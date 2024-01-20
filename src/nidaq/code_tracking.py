from typing import Optional
import sys
import inspect
import logging
import contextlib
from os import PathLike
from pathlib import Path
from datetime import datetime
import git
import subprocess

class CodeTrackingError(Exception):
    pass

def getRepo(filePath: PathLike) -> git.Repo | CodeTrackingError:
    filePath = Path(filePath)
    try:
        repo = git.Repo(filePath.parent, search_parent_directories = True)
    except git.InvalidGitRepositoryError as e:
        return CodeTrackingError("Script is not in a valid git repo for tracking.")
    if repo.bare:
        return CodeTrackingError("Script repo cannot be bare.")
    return repo

def runGitCommand(repo: git.Repo, argv: list[str]):
    with contextlib.chdir(repo.working_tree_dir):
        return subprocess.run(["git", *argv])

def workingTreeState(repo: git.Repo, startTime: datetime, no_untracked = True, dirtyOK = False, autocommit = True) -> tuple[Optional[str] | CodeTrackingError, str, bool]:
    with contextlib.chdir(repo.working_tree_dir):
        description = repo.git.describe("--long", "--dirty")
        if not repo.is_dirty(untracked_files = no_untracked):
            return (repo.head.commit.hexsha, description, False)

        try:
            # Run our own git subprocess so that we get pretty colors
            runGitCommand(repo, ["status", "-s", "-b"])
        except FileNotFoundError:
            repo.git.status("-s", "-b")

        if dirtyOK:
            return (None, description, True)

        if not autocommit:
            return (CodeTrackingError("Working tree is dirty and autocommit is disabled."), description, True)

        try:
            timestamp = startTime.astimezone().isoformat()
            author = "Alex Striff <abs299@cornell.edu>" # TODO: Conveniently configure this for different users? Look for session config?
            message = f"{timestamp} : first execution for new tree"
            runGitCommand(repo, ["add", "--all"])
            runGitCommand(repo, ["commit", "--author", author, "-m", message])
            return (repo.head.commit.hexsha, description, True)
        except:
            return (CodeTrackingError(f"There was an error while autocommiting repo at {Path(repo.working_tree_dir).resolve()}"), description, True)

def fileExecutionData(
        filePath: PathLike,
        argv: list[str],
        startTime: datetime = None,
        dirtyOK: bool = False,
        ) -> dict | CodeTrackingError:
    filePath = Path(filePath)
    if startTime is None:
        startTime = datetime.now()
    repo = getRepo(filePath)
    if isinstance(repo, CodeTrackingError):
        return repo
    repoWorkingTree = repo.working_tree_dir

    commitResult, description, wasDirty = workingTreeState(
            repo,
            startTime = startTime,
            dirtyOK = dirtyOK,
            )
    if isinstance(commitResult, CodeTrackingError):
        # return commitResult
        raise commitResult

    return {
        "file": filePath.resolve().relative_to(repoWorkingTree).as_posix(),
        "argv": [Path(argv[0]).as_posix(), *argv[1:]],
        "timestamp": startTime.astimezone().isoformat(),
        "git_repo": {
            "working_tree_description": description,
            "commit": commitResult if commitResult is not None else f"<dirty>",
            "ref": repo.head.ref.name,
            "remotes": {remote.name: list(remote.urls) for remote in repo.remotes},
            "working_tree": Path(repoWorkingTree).as_posix(),
        }
    }

if __name__ == "__main__":
    import json

    x = fileExecutionData(__file__, sys.argv)
    if isinstance(x, CodeTrackingError):
        print(x)
        sys.exit(1)

    j = json.dumps(x, indent = 2)
    print(j)


