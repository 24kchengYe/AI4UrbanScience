from pathlib import Path

from scripts import build_manifest
from scripts import verify_release


def test_residue_scan_allows_only_root_git_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    clone_root = tmp_path / "clone"
    (clone_root / ".git" / "objects" / "build").mkdir(parents=True)
    (clone_root / ".git" / "objects" / "cache.pyc").write_bytes(b"fixture")
    (clone_root / "public.txt").write_text("public\n", encoding="utf-8")
    monkeypatch.setattr(verify_release, "ROOT", clone_root)
    assert verify_release._generated_residue_paths() == []
    assert [path.relative_to(clone_root).as_posix() for path in verify_release.repository_files()] == ["public.txt"]

    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()
    (worktree_root / ".git").write_text(
        "gitdir: ../repository/.git/worktrees/fixture\n", encoding="utf-8"
    )
    (worktree_root / "public.txt").write_text("public\n", encoding="utf-8")
    monkeypatch.setattr(verify_release, "ROOT", worktree_root)
    assert verify_release._generated_residue_paths() == []
    assert [path.relative_to(worktree_root).as_posix() for path in verify_release.repository_files()] == ["public.txt"]
    monkeypatch.setattr(build_manifest, "ROOT", worktree_root)
    assert [path.relative_to(worktree_root).as_posix() for path in build_manifest.included_files()] == ["public.txt"]

    residue_root = tmp_path / "residue"
    for name in verify_release.GENERATED_DIRECTORIES:
        (residue_root / name).mkdir(parents=True, exist_ok=True)
    (residue_root / "package.egg-info").mkdir()
    (residue_root / "nested" / ".git").mkdir(parents=True)
    (residue_root / "cache.pyc").write_bytes(b"fixture")
    (residue_root / "cache.pyo").write_bytes(b"fixture")
    monkeypatch.setattr(verify_release, "ROOT", residue_root)

    observed = set(verify_release._generated_residue_paths())
    expected = set(verify_release.GENERATED_DIRECTORIES) | {
        "package.egg-info",
        "nested/.git",
        "cache.pyc",
        "cache.pyo",
    }
    assert expected.issubset(observed)
