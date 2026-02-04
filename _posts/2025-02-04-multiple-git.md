---
layout: post
title: "Multiple git account 깃 복수계정"
date: 2025-02-04
tags: [git]
---

The cleanest ways to use **multiple Git accounts locally** are usually these two:

1. **Separate SSH keys per account** (recommended — the most reliable)
2. Use different **user.name / user.email per repository** (this only changes the commit author — push permissions/auth are separate)

Below is a standard setup for the case where **both Account A and Account B are GitHub accounts**, fully separated using SSH.

---

## 1) Generate SSH keys for each account

```bash
ssh-keygen -t ed25519 -C "accountA@email.com" -f ~/.ssh/id_ed25519_github_a
ssh-keygen -t ed25519 -C "accountB@email.com" -f ~/.ssh/id_ed25519_github_b
```

---

## 2) Create host aliases in `~/.ssh/config`

```sshconfig
Host github-a
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github_a
  IdentitiesOnly yes

Host github-b
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github_b
  IdentitiesOnly yes
```

---

## 3) Add the public keys to each GitHub account

Copy each public key and add it in GitHub → **Settings → SSH and GPG keys**.

```bash
cat ~/.ssh/id_ed25519_github_a.pub
cat ~/.ssh/id_ed25519_github_b.pub
```

---

## 4) Test the connection

```bash
ssh -T git@github-a
ssh -T git@github-b
```

If you see “Hi username!” for each one, it worked.

---

## 5) Decide which account the repo should use (clone/connect)

For example, if you want to push using Account B:

### When cloning a new repo

```bash
git clone git@github-b:USERNAME/REPO.git
```

### If the repo is already cloned, just change the remote URL

```bash
git remote set-url origin git@github-b:USERNAME/REPO.git
```

---

## 6) Separate commit author per repository (optional)

Inside the repo folder:

```bash
git config user.name "AccountB Name"
git config user.email "accountB@email.com"
```

Check:

```bash
git config user.name
git config user.email
git remote -v
```

---

## Key takeaway for a GitHub Pages repo

* **Push authentication (permissions/login)** is determined by whether `origin` uses `git@github-a:` or `git@github-b:`
* **The email shown in commits** is determined by `git config user.email`

