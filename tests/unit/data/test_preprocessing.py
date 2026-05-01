"""Unit tests for the data preprocessing layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.preprocessing.cleaner import TextCleaner
from src.data.preprocessing.formatter import ChatMLFormatter
from src.data.preprocessing.splitter import DatasetSplitter


# ── TextCleaner ───────────────────────────────────────────────────────────────


class TestTextCleaner:
    _CLEANER = TextCleaner()

    def test_removes_control_chars(self) -> None:
        text = "Hello\x00\x01World"
        result = self._CLEANER.clean(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "HelloWorld" in result

    def test_fixes_pdf_hyphenation(self) -> None:
        text = "regu-\nlation"
        result = self._CLEANER.clean(text)
        assert "regulation" in result

    def test_collapses_multiple_spaces(self) -> None:
        text = "Hello   world"
        result = self._CLEANER.clean(text)
        assert "Hello world" in result

    def test_preserves_devanagari(self) -> None:
        text = "आयकर अधिनियम 1961"
        result = self._CLEANER.clean(text)
        assert "आयकर" in result

    def test_removes_noise_lines(self) -> None:
        text = "Real content\n\nX\n\nMore content"
        result = TextCleaner(min_line_length=3).clean(text)
        assert "X\n" not in result

    def test_collapses_blank_lines(self) -> None:
        text = "Line 1\n\n\n\n\nLine 2"
        result = TextCleaner(max_consecutive_newlines=2).clean(text)
        assert "\n\n\n" not in result

    def test_empty_string(self) -> None:
        result = self._CLEANER.clean("")
        assert result == ""


# ── ChatMLFormatter ───────────────────────────────────────────────────────────


class TestChatMLFormatter:
    _FORMATTER = ChatMLFormatter()

    def test_from_qa_dict(self) -> None:
        qa = {"question": "What is 80C?", "answer": "Section 80C allows deductions..."}
        result = self._FORMATTER.from_qa_dict(qa)
        assert "messages" in result
        roles = {m["role"] for m in result["messages"]}
        assert roles == {"system", "user", "assistant"}

    def test_from_qa_dict_missing_key_raises(self) -> None:
        with pytest.raises(ValueError):
            self._FORMATTER.from_qa_dict({"question": "only question"})

    def test_system_prompt_in_output(self) -> None:
        qa = {"question": "Q", "answer": "A" * 50}
        result = self._FORMATTER.from_qa_dict(qa)
        system_msg = next(m for m in result["messages"] if m["role"] == "system")
        assert "FinEdge" in system_msg["content"]

    def test_custom_system_prompt(self) -> None:
        formatter = ChatMLFormatter(system_prompt="Custom prompt")
        qa = {"question": "Q?", "answer": "Answer here."}
        result = formatter.from_qa_dict(qa)
        system_msg = next(m for m in result["messages"] if m["role"] == "system")
        assert system_msg["content"] == "Custom prompt"

    def test_validate_chatml_valid(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        assert self._FORMATTER.validate_chatml(sample) is True

    def test_validate_chatml_missing_role(self) -> None:
        sample = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        assert self._FORMATTER.validate_chatml(sample) is False

    def test_cleaner_applied_to_content(self) -> None:
        qa = {"question": "What   is   80C?", "answer": "Section\x00 80C allows..."}
        result = self._FORMATTER.from_qa_dict(qa)
        user_content = next(m["content"] for m in result["messages"] if m["role"] == "user")
        assert "  " not in user_content


# ── DatasetSplitter ───────────────────────────────────────────────────────────


class TestDatasetSplitter:
    _SAMPLES = [{"messages": [{"role": "user", "content": f"Q{i}"}]} for i in range(100)]

    def test_split_sizes_correct(self) -> None:
        splitter = DatasetSplitter(train_ratio=0.8, eval_ratio=0.1)
        train, eval_, test = splitter.split(self._SAMPLES)
        assert len(train) == 80
        assert len(eval_) == 10
        assert len(test) == 10

    def test_no_overlap_between_splits(self) -> None:
        splitter = DatasetSplitter()
        train, eval_, test = splitter.split(self._SAMPLES)
        train_set = {id(s) for s in train}
        eval_set = {id(s) for s in eval_}
        test_set = {id(s) for s in test}
        assert not train_set & eval_set
        assert not train_set & test_set
        assert not eval_set & test_set

    def test_all_samples_accounted_for(self) -> None:
        splitter = DatasetSplitter()
        train, eval_, test = splitter.split(self._SAMPLES)
        assert len(train) + len(eval_) + len(test) == len(self._SAMPLES)

    def test_deterministic_with_same_seed(self) -> None:
        s1 = DatasetSplitter(seed=42)
        s2 = DatasetSplitter(seed=42)
        t1, e1, _ = s1.split(self._SAMPLES)
        t2, e2, _ = s2.split(self._SAMPLES)
        assert t1 == t2
        assert e1 == e2

    def test_different_seed_gives_different_split(self) -> None:
        s1 = DatasetSplitter(seed=1)
        s2 = DatasetSplitter(seed=2)
        t1, _, _ = s1.split(self._SAMPLES)
        t2, _, _ = s2.split(self._SAMPLES)
        assert t1 != t2

    def test_invalid_ratios_raise(self) -> None:
        with pytest.raises(ValueError):
            DatasetSplitter(train_ratio=0.9, eval_ratio=0.2)

    def test_save_creates_files(self, tmp_path: Path) -> None:
        splitter = DatasetSplitter()
        train_dir = tmp_path / "train"
        eval_dir = tmp_path / "eval"
        test_dir = tmp_path / "test"
        splitter.save(self._SAMPLES, train_dir, eval_dir, test_dir)
        assert (train_dir / "data.jsonl").exists()
        assert (eval_dir / "data.jsonl").exists()
        assert (test_dir / "data.jsonl").exists()

    def test_load_jsonl_round_trip(self, tmp_path: Path) -> None:
        import json

        out = tmp_path / "test.jsonl"
        samples = [{"messages": [{"role": "user", "content": f"Q{i}"}]} for i in range(5)]
        with out.open("w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        loaded = DatasetSplitter.load_jsonl(out)
        assert len(loaded) == 5
        assert loaded[0]["messages"][0]["content"] == "Q0"

    def test_load_jsonl_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetSplitter.load_jsonl(tmp_path / "nonexistent.jsonl")
