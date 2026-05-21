# ============================================================
# GitHub Config
# ============================================================

import os
import requests
import base64

# --------------------------------------------------
# Secrets
# --------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("Missing GITHUB_TOKEN environment variable")

# --------------------------------------------------
# Repo Config
# --------------------------------------------------
OWNER = "StarpowerTechnology"
REPO = "SavvyResearcher"
BRANCH = "main"
API = "https://api.github.com"

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

print("✅ Connected to GitHub")
print(f"Repo: {OWNER}/{REPO}")


# ============================================================
# Helpers
# ============================================================

def get_contents(path=""):
    url = f"{API}/repos/{OWNER}/{REPO}/contents/{path}"
    return requests.get(url, headers=HEADERS, params={"ref": BRANCH})


def get_sha(path):
    r = get_contents(path)
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, dict):
            return data.get("sha")
    return None


# ============================================================
# View FULL Repo Tree (All folders + files)
# ============================================================

def view_file_list():
    url = f"{API}/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"
    r = requests.get(url, headers=HEADERS)

    if r.status_code != 200:
        print("❌ Failed to load repo tree")
        print(r.text)
        return

    data = r.json()

    print("\n📂 FULL REPO STRUCTURE")
    print("--------------------------------------------------")

    for item in data["tree"]:
        if item["type"] == "tree":
            print(f"📁 {item['path']}/")
        else:
            print(f"📄 {item['path']}")

    print("--------------------------------------------------\n")


# ============================================================
# Read File
# ============================================================

def read_file(path):
    r = get_contents(path)

    if r.status_code != 200:
        print(f"❌ Failed to read file: {path}")
        print(r.text)
        return

    data = r.json()

    if data["type"] != "file":
        print("❌ Not a file")
        return

    content = base64.b64decode(data["content"]).decode("utf-8")

    print(f"\n📄 FILE: {path}")
    print("--------------------------------------------------")
    print(content)
    print("--------------------------------------------------\n")


# ============================================================
# Create File
# ============================================================

def create_file(path, message, content):
    if get_sha(path):
        print(f"❌ File already exists: {path}")
        return

    url = f"{API}/repos/{OWNER}/{REPO}/contents/{path}"

    payload = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": BRANCH
    }

    r = requests.put(url, headers=HEADERS, json=payload)

    if r.status_code in (200, 201):
        print(f"✅ Created file: {path}")
    else:
        print(f"❌ Failed to create file: {path}")
        print(r.text)


# ============================================================
# Update File
# ============================================================

def update_file(path, message, content):
    sha = get_sha(path)

    if not sha:
        print(f"❌ File not found: {path}")
        return

    url = f"{API}/repos/{OWNER}/{REPO}/contents/{path}"

    payload = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "sha": sha,
        "branch": BRANCH
    }

    r = requests.put(url, headers=HEADERS, json=payload)

    if r.status_code in (200, 201):
        print(f"✅ Updated file: {path}")
    else:
        print(f"❌ Failed to update file: {path}")
        print(r.text)


# ============================================================
# Delete File
# ============================================================

def delete_file(path, message):
    sha = get_sha(path)

    if not sha:
        print(f"❌ File not found: {path}")
        return

    url = f"{API}/repos/{OWNER}/{REPO}/contents/{path}"

    payload = {
        "message": message,
        "sha": sha,
        "branch": BRANCH
    }

    r = requests.delete(url, headers=HEADERS, json=payload)

    if r.status_code in (200, 201):
        print(f"✅ Deleted file: {path}")
    else:
        print(f"❌ Failed to delete file: {path}")
        print(r.text)


# ============================================================
# Create Folder
# ============================================================

def create_folder(folder):
    path = f"{folder}/.gitkeep"
    create_file(path, f"Create folder {folder}", "")
