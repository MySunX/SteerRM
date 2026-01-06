import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI


DOMAIN_PROMPTS = {
    "code": """Generate a unique programming-related question (such as algorithm implementation, code explanation, debugging, etc.).
Be creative with the specific topic, programming language, and difficulty level.""",
    "math": """Generate a unique mathematics question (such as geometry, algebra, calculus, etc.).
Be creative with the specific topic and difficulty level.""",
    "chat": """Generate a unique general conversation question (such as life advice, knowledge Q&A, etc.).
Be creative with the specific topic.""",
    "reasoning": """Generate a unique reasoning question (such as logical reasoning, causal analysis, etc.).
Be creative with the specific topic.""",
}


class JsonArrayStreamWriter:
    def __init__(self, path: Path, *, append: bool = False, existing_items: int = 0):
        self.path = path
        self._append = append
        self._existing_items = existing_items
        self._first = True
        self._closed = False

        if not append:
            self._fp = path.open("w", encoding="utf-8")
            self._fp.write("[\n")
            self._fp.flush()
            os.fsync(self._fp.fileno())
            self._first = True
            return

        if not path.exists() or path.stat().st_size == 0:
            self._fp = path.open("w", encoding="utf-8")
            self._fp.write("[\n")
            self._fp.flush()
            os.fsync(self._fp.fileno())
            self._first = True
            return

        with path.open("rb+") as bfp:
            bfp.seek(0, os.SEEK_END)
            end = bfp.tell()
            i = end - 1
            while i >= 0:
                bfp.seek(i)
                ch = bfp.read(1)
                if ch not in b" \t\r\n":
                    break
                i -= 1
            if i < 0:
                bfp.truncate(0)
            else:
                if ch != b"]":
                    raise ValueError(f"Existing file is not a JSON array ending with ']': {path}")
                bfp.truncate(i)

        self._fp = path.open("r+", encoding="utf-8")
        self._fp.seek(0, os.SEEK_END)
        self._first = (existing_items == 0)
        if self._first:
            try:
                with path.open("rb") as rb:
                    rb.seek(0, os.SEEK_END)
                    end2 = rb.tell()
                    j = end2 - 1
                    last_non_ws = None
                    while j >= 0:
                        rb.seek(j)
                        ch2 = rb.read(1)
                        if ch2 not in b" \t\r\n":
                            last_non_ws = ch2
                            break
                        j -= 1
                if last_non_ws == b"[":
                    self._fp.write("\n")
            except Exception:
                pass
        self._fp.flush()
        os.fsync(self._fp.fileno())

    def append(self, obj: Dict) -> None:
        if self._closed:
            raise RuntimeError("Writer is closed")
        if not self._first:
            self._fp.write(",\n")
        self._fp.write(json.dumps(obj, ensure_ascii=False, indent=2))
        self._first = False
        self._fp.flush()
        os.fsync(self._fp.fileno())

    def close(self) -> None:
        if self._closed:
            return
        self._fp.write("\n]\n")
        self._fp.flush()
        os.fsync(self._fp.fileno())
        self._fp.close()
        self._closed = True


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _robust_json_loads(text: Optional[str]) -> Dict:
    if text is None:
        raise ValueError("Empty response content (None)")
    t = _strip_code_fences(str(text))
    if not t.strip():
        raise ValueError(f"Empty response content: {repr(text)[:120]}")

    try:
        obj = json.loads(t)
        if not isinstance(obj, dict):
            raise ValueError("JSON root is not an object")
        return obj
    except Exception:
        pass

    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start : end + 1]
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            raise ValueError("JSON root is not an object")
        return obj

    raise ValueError("Unable to locate a JSON object in response")


def _atomic_write_json_array(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _normalize_existing_ids(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        existing_data = json.load(f)
    if not isinstance(existing_data, list):
        raise ValueError("Existing JSON root is not an array")

    changed = 0
    for i, item in enumerate(existing_data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} is not a JSON object; cannot set id safely.")
        if item.get("id") != i:
            item["id"] = i
            changed += 1

    if changed > 0:
        _atomic_write_json_array(path, existing_data)
    return changed


def llm_generate_sample(
    client: OpenAI,
    model: str,
    domain: str,
    sample_id: int,
) -> Dict:
    domain_instruction = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["chat"])
    
    system_prompt = """You are a data synthesis assistant responsible for generating training data.

Task (two steps):
1) Generate a unique question (prompt) and an answer written in Markdown (answer_markdown).
2) Remove Markdown formatting from answer_markdown to generate a plain-text version (answer_plain).

Requirements:
- Use Markdown naturally for readability. Start the answer with a normal sentence (do not begin with a heading); you may use headings later if helpful.
- answer_plain must preserve the exact wording/content of answer_markdown.
- Only remove Markdown syntax/markup tokens. Do NOT rewrite, paraphrase, add, or delete content.

Output format as JSON:
{
  "prompt": "...",
  "answer_markdown": "...",
  "answer_plain": "..."
}"""

    user_prompt = f"""{domain_instruction}

Generate ONE complete training sample. Requirements recap:
- prompt: clear, specific, unambiguous.
- answer_markdown: a natural assistant answer written in Markdown.
- answer_plain: plain text version of answer_markdown with ALL Markdown removed; keep the exact same wording/content.
- Do not add/omit content between the two.

Output only JSON, no other content."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
            timeout=60.0,
        )
    except Exception as e:
        raise RuntimeError(f"API call failed for sample {sample_id}: {str(e)}") from e

    try:
        choice0 = resp.choices[0]
        msg = getattr(choice0, "message", None)
        content = getattr(msg, "content", None)
        finish_reason = getattr(choice0, "finish_reason", None)
        role = getattr(msg, "role", None)
    except Exception:
        content = None
        finish_reason = None
        role = None

    if content is None or (isinstance(content, str) and not content.strip()):
        raise RuntimeError(
            "Empty response content from server. "
            f"finish_reason={finish_reason}, role={role}, content_len={0 if content is None else len(content)}"
        )

    try:
        data = _robust_json_loads(content)
    except Exception as exc:
        preview = "" if content is None else str(content)[:300]
        raise RuntimeError(f"Unable to parse JSON. Response content preview: {preview}") from exc
    
    try:
        required_fields = ["prompt", "answer_markdown", "answer_plain"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            if not data[field] or not isinstance(data[field], str):
                raise ValueError(f"{field} cannot be empty")
        
        data["id"] = sample_id
        data["subset"] = f"synthetic-{domain}"
        
        return data
        
    except json.JSONDecodeError as exc:
        preview = "" if content is None else str(content)[:300]
        raise RuntimeError(f"Unable to parse JSON. Response content preview: {preview}") from exc
    except Exception as exc:
        preview = "" if content is None else str(content)[:300]
        raise RuntimeError(f"Data validation failed: {str(exc)}. Content preview: {preview}") from exc


def generate_dataset(
    n: int,
    out_dir: str,
    model: str,
    domain: str = "mixed",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_workers: int = 10,
    max_retries: int = 5
) -> Path:
    client = OpenAI(api_key=api_key, base_url=base_url)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"markdown_dataset_{domain}.json"

    existing_count = 0
    if out_file.exists() and out_file.stat().st_size > 0:
        try:
            changed = _normalize_existing_ids(out_file)
            with out_file.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_count = len(existing_data) if isinstance(existing_data, list) else 0
            if changed > 0:
                print(f"Normalized existing ids in {out_file}: fixed {changed} item(s).")
        except Exception as e:
            raise RuntimeError(
                f"Existing output file is not a valid JSON array, cannot resume safely: {out_file}. "
                f"Please fix or remove it. Error: {e}"
            ) from e

    if existing_count >= n:
        print(f"Found existing dataset with {existing_count} items at {out_file}. Nothing to do (n={n}).")
        return out_file

    writer = JsonArrayStreamWriter(
        out_file,
        append=(out_file.exists() and out_file.stat().st_size > 0),
        existing_items=existing_count,
    )

    samples_written = existing_count
    sample_id_counter = existing_count
    
    domains = list(DOMAIN_PROMPTS.keys()) if domain == "mixed" else [domain]
    
    if existing_count > 0:
        print(f"Found existing dataset with {existing_count} items at {out_file}. Will append to reach {n}.")
    print(f"Starting to generate {n} samples (domain: {domain}) with {max_workers} parallel workers...")
    print(f"Will retry failed samples up to {max_retries} times")
    
    def generate_single_sample(sample_id):
        current_domain = random.choice(domains) if domain == "mixed" else domain
        try:
            sample = llm_generate_sample(client, model, current_domain, sample_id)
            return (True, sample, current_domain, None)
        except Exception as e:
            return (False, None, current_domain, str(e))
    
    retry_count = 0
    total_attempts = 0
    failed_samples = []

    try:
        while samples_written < n and retry_count < max_retries:
            needed = n - samples_written

            if retry_count > 0:
                print(f"\nRetry {retry_count}/{max_retries}: Need {needed} more samples...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                start_id = sample_id_counter
                futures = {
                    executor.submit(generate_single_sample, start_id + i): start_id + i
                    for i in range(needed)
                }
                sample_id_counter += needed

                desc = f"Generating (attempt {retry_count + 1})" if retry_count > 0 else "Generating samples"
                with tqdm(total=needed, desc=desc) as pbar:
                    for future in as_completed(futures):
                        total_attempts += 1
                        success, result, current_domain, error = future.result()
                        if success:
                            result["id"] = samples_written
                            writer.append(result)
                            samples_written += 1
                            pbar.set_postfix({"status": f"✓ {current_domain}", "total": samples_written})
                        else:
                            failed_samples.append({"domain": current_domain, "error": error})
                            pbar.set_postfix({"status": f"✗ Error", "total": samples_written})
                            if len(failed_samples) <= 3:
                                try:
                                    pbar.clear()
                                    pbar.write(f"[ERROR] Sample failed: {error[:500]}")
                                    pbar.refresh()
                                except Exception:
                                    print(f"[ERROR] Sample failed: {error[:500]}", file=sys.stderr)
                        pbar.update(1)

            if samples_written < n:
                retry_count += 1
                print(f"Generated {samples_written}/{n} samples, will retry for remaining {n - samples_written} samples...")
            else:
                break
    finally:
        writer.close()

    print(f"\n{'='*60}")
    print(f"Successfully generated {samples_written}/{n} samples")
    print(f"Total attempts: {total_attempts}")
    print(f"Success rate: {samples_written/total_attempts*100:.1f}%" if total_attempts else "Success rate: N/A")
    print(f"Failed samples: {len(failed_samples)}")
    if failed_samples:
        print(f"\nError summary (showing up to 5):")
        for i, fail in enumerate(failed_samples[:5]):
            print(f"  {i+1}. [{fail['domain']}] {fail['error'][:150]}...")
    print(f"Saved to: {out_file}")
    print(f"{'='*60}")

    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown/Plain paired dataset using LLM"
    )
    parser.add_argument("--n", type=int, default=50, help="Number of samples to generate")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name to use")
    parser.add_argument(
        "--domain",
        type=str,
        default="mixed",
        choices=["code", "math", "chat", "reasoning", "mixed"],
        help="Data domain (mixed means all domains combined)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key (can be left empty to use OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom API base URL"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel workers for generation"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retry attempts for failed samples"
    )
    args = parser.parse_args()
    
    out_file = generate_dataset(
        args.n,
        args.out_dir,
        args.model,
        args.domain,
        args.api_key,
        args.base_url,
        args.max_workers,
        args.max_retries
    )


if __name__ == "__main__":
    main()

