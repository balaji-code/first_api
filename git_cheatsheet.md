🧭 Git + API Development — Solo Dev Cheatsheet

(1-page guide; keep pinned in Cursor)

⸻

🔑 Core Principles
	•	Never work directly on main
	•	Always use feature branches
	•	Test locally before committing
	•	Commit small, logical chunks
	•	Push branches early (backup)
	•	Merge only after testing

⸻

🚀 Standard Feature Workflow
1️⃣ Create a new feature branch
git checkout -b feat/<short-description>
# e.g., feat/sentiment-endpoint

2️⃣ Code & test locally

Run the server:
python -u app.py

3️⃣ Stage only intentional changes
git add <file1> <file2>

4️⃣ Commit (Conventional style)
git commit -m "feat(sentiment): add sentiment endpoint with label+score"

5️⃣ Push branch to GitHub
git push -u origin feat/<short-description>

6️⃣ Merge into main

Option A — GitHub UI (recommended)
Open PR → Review yourself → Merge

Option B — Terminal merge
git checkout main
git pull origin main
git merge --no-ff feat/<short-description>
git push origin main
git branch -d feat/<short-description>
✔ Pre-Merge Checklist
	•	Server starts with no errors
	•	Endpoint works for happy + unhappy paths
	•	git status clean
	•	Only intended files are staged
	•	Commit message clear
	•	Requirements updated (if needed)

🛠 If Something Breaks (Solo Dev Fix)

Hotfix (safe & best)
git checkout -b hotfix/<name> main
# fix & test
git add .
git commit -m "fix: correct parsing logic"
git push -u origin hotfix/<name>
# merge hotfix into main


Revert a bad commit (safe for remote)
git revert <commit-hash>
git push origin main
Reset & force (ONLY solo projects)
git checkout main
git reset --hard <previous-good-commit>
git push --force origin main

📝 Commit Message Templates
	•	feat(<scope>): ... → new feature
	•	fix(<scope>): ... → bug fix
	•	docs: ... → documentation changes
	•	refactor: ... → internal cleanups
	•	chore: ... → dependency or config updates

Examples:
feat(summarize): add improved bullet formatting
fix(keywords): handle empty list from model
docs: add API usage examples to README

alias gco='git checkout'
alias gcb='git checkout -b'
alias gst='git status'
alias gpo='git push -u origin'
alias gpl='git pull origin main'

⚡ Optional Zsh Aliases (add to ~/.zshrc)

🧠 Final Rules (non-negotiable)
	•	Don’t push broken code to main
	•	Don’t force-push main unless absolutely alone and certain
	•	Keep branches focused, small, and testable
	•	Use PRs even when solo (they act as your history)

# Start fresh
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/my-task

# Work on your code ...
# Then stage + commit
git add .
git commit -m "Finish my feature"

# Push to GitHub
git push origin feature/my-task

# Create PR → Merge on GitHub

# Update local main
git checkout main
git pull origin main

# Delete merged branch locally
git branch -d feature/my-task

# Delete branch from GitHub (optional)
git push origin --delete feature/my-task

# important curl commands
for id in $(curl -s http://127.0.0.1:5050/rag-docs | jq '.chunks[] | select(.metadata._parent=="mixed-1") | .id'); do
  echo "Deleting chunk ID: $id"
  curl -s -X POST http://127.0.0.1:5050/rag-delete -H "Content-Type: application/json" --data "{\"id\":$id}"
  echo
done
