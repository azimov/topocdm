# Mirroring to a separate work account

The working repository stays private under the personal account. The work
account gets a **private mirror** so the work scanner can see it; the public
never does.

This is a two-remote setup, not a visibility change. Do not make either
repository public.

## One-time setup

Use the work account's credentials for the creation step (or create the repo
in the GitHub web UI while signed in as the work account).

```bash
# signed in as the work account
gh repo create <work-account-or-org>/topocdm --private

# from the personal checkout
git remote add work git@github.com:<work-account-or-org>/topocdm.git
# or, with HTTPS:
# git remote add work https://github.com/<work-account-or-org>/topocdm.git

git push -u work main
```

If the employer requires the repository to be owned by the organisation rather
than the personal work account, use `<work-org>` as the owner. The repository
is still private.

## Verify it is private

```bash
gh repo view <work-account-or-org>/topocdm --json visibility,nameWithOwner
# expect: "visibility": "PRIVATE"
```

Check the remotes:

```bash
git remote -v
# origin  .../<personal-account>/topocdm.git   (private, personal)
# work    .../<work-account-or-org>/topocdm.git (private, work)
```

## Day-to-day

Push to both when you want work to see the latest:

```bash
git push origin main
git push work main
```

or use the guarded script:

```bash
bash scripts/push_work.sh
```

It refuses to push if the working tree is dirty or if any tracked file matches
the data/secret patterns (`.secrets/`, `data/`, `*.pdf`, `*.pkl`, `*.env`).

## Rules

- **Neither repository is public.** The work mirror exists so the scanner can
  see the work; it is not a publication.
- **The handoff and strategy documents are in the mirror.** That is
  intentional: the work account is allowed to see them. The world is not.
- **If you later want a public method repository, do a separate stripped
  export.** Do not unprivate the work mirror. A public repo is permanent in
  practice (forks, clones, search caches), and it would expose the handoff,
  the graded claims, and the negative-result notes.
- **Do not put the work token in the repository.** Use `gh auth login`, SSH
  keys, or the operating system's credential helper.
- **Do not add `work` as the default push remote.** Keep `origin` personal and
  push to `work` deliberately.
