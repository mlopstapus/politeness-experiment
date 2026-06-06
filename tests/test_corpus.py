import json
from pathlib import Path

CORPUS_PATH = Path("corpus/tasks.json")


def load_corpus():
    with CORPUS_PATH.open() as f:
        return json.load(f)


def test_corpus_exists():
    assert CORPUS_PATH.exists()


def test_corpus_has_30_tasks():
    assert len(load_corpus()) == 30


def test_each_task_has_required_fields():
    for task in load_corpus():
        assert "task_id" in task
        assert "title" in task
        assert "content" in task
        assert "topic" in task


def test_task_ids_are_unique_and_sequential():
    corpus = load_corpus()
    ids = sorted(t["task_id"] for t in corpus)
    assert ids == [f"T{i:02d}" for i in range(1, 31)]


def test_content_word_count_in_range():
    for task in load_corpus():
        wc = len(task["content"].split())
        assert 200 <= wc <= 900, f"{task['task_id']} has {wc} words"


def test_no_duplicate_first_sentences():
    corpus = load_corpus()
    first = [t["content"].split(".")[0] for t in corpus]
    assert len(first) == len(set(first)), "Duplicate first sentences — cache collision risk"


def test_ten_topic_areas():
    assert len(set(t["topic"] for t in load_corpus())) == 10
