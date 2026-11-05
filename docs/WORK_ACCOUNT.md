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

## If the work account is an enterprise-managed user (EMU)

A username with an enterprise suffix (for example `JGilber2_JNJT`) is usually
an enterprise-managed user. EMU accounts cannot be added as outside
collaborators to repositories owned by personal accounts, cannot join outside
organisations, and cannot fork outside repositories. The "share" or collaborator
route therefore fails, and no setting on the personal repository fixes it.

The mirror must live inside the work enterprise:

1. Sign in as the work account and create a **private** repository `topocdm`
   in the enterprise (under the EMU account or a work organisation). Copy its
   URL.
2. On the personal machine, authenticate as the work account for that remote.
   SSH is the cleanest:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_work -C "work laptop"
   # add ~/.ssh/id_ed25519_work.pub to the work account's SSH keys
   ```

   Add a host alias to `~/.ssh/config`:

   ```
   Host github-work
     HostName github.com
     User git
     IdentityFile ~/.ssh/id_ed25519_work
     IdentitiesOnly yes
   ```

   Then add the remote through the alias:

   ```bash
   git remote add work git@github-work:<owner>/topocdm.git
   ```

   A work PAT with `contents: write` and a credential helper also works. Do
   not put the token in the remote URL or in the repository.

3. Push from the personal machine:

   ```bash
   git push work main
   # or: bash scripts/push_work.sh
   ```

The personal machine is pushing into the enterprise; the EMU account does not
need access to the personal repository. If the enterprise blocks pushes from
personal devices, run the same push from the work laptop after cloning the
work repository there.

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
