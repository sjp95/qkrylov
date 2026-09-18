import os
import sys
import json
import urllib.request
from collections import defaultdict


def generate_index_html(package_name, wheel_data, repo_slug):
    """
    Generates a PEP 503 compliant Simple Repository API HTML page.
    Links point to GitHub Releases download URLs.
    """
    html = (
        f"<!DOCTYPE html>\n<html>\n<head>\n"
        f"  <title>Links for {package_name}</title>\n"
        f"</head>\n<body>\n"
        f"  <h1>Links for {package_name}</h1>\n"
    )

    # Deduplicate by wheel filename (first occurrence wins)
    seen = set()
    unique_wheels = []
    for tag, wheel in wheel_data:
        if wheel not in seen:
            seen.add(wheel)
            unique_wheels.append((tag, wheel))

    unique_wheels.sort(key=lambda x: x[1])

    for tag, wheel in unique_wheels:
        url = f"https://github.com/{repo_slug}/releases/download/{tag}/{wheel}"
        html += f'  <a href="{url}">{wheel}</a><br/>\n'

    html += "</body>\n</html>\n"
    return html


def wheel_variant(wheel):
    parts = wheel.split("-")
    if len(parts) < 2:
        return None
    version = parts[1]
    if "+" not in version:
        return "cpu"
    return version.split("+", 1)[1]


def get_historical_wheels(repo_slug, exclude_tag=None):
    """
    Fetch wheel assets from ALL GitHub Releases (paginated).
    Optionally exclude a specific tag (e.g. the current release being built,
    whose assets may still be uploading when this runs).
    """
    all_assets = []
    page = 1

    while True:
        url = (
            f"https://api.github.com/repos/{repo_slug}/releases"
            f"?per_page=100&page={page}"
        )
        req = urllib.request.Request(url)
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")

        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            print(f"Warning: Failed to fetch releases page {page}: {e}")
            break

        if not data:
            break

        for release in data:
            tag = release.get("tag_name")
            if tag == exclude_tag:
                continue
            for asset in release.get("assets", []):
                name = asset.get("name", "")
                if name.endswith(".whl"):
                    all_assets.append((tag, name))

        # GitHub returns fewer items than per_page when on last page
        if len(data) < 100:
            break

        page += 1

    return all_assets


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: generate_indexes.py <wheel_dir> <repo_slug> <tag>")
        sys.exit(1)

    wheel_dir = sys.argv[1]
    repo_slug = sys.argv[2]
    current_tag = sys.argv[3]
    package_name = "qkrylov"

    # 1. Local wheels from the current workflow run (authoritative for current_tag)
    local_wheels = [f for f in os.listdir(wheel_dir) if f.endswith(".whl")]
    all_wheels = [(current_tag, w) for w in local_wheels]
    print(f"Found {len(local_wheels)} local wheels for {current_tag}.")

    # 2. Historical wheels from GitHub API (excludes current_tag to avoid race)
    historical = get_historical_wheels(repo_slug, exclude_tag=current_tag)
    all_wheels.extend(historical)
    print(f"Found {len(historical)} historical wheels from past releases.")

    if not all_wheels:
        print("No wheels found locally or historically.")
        sys.exit(0)

    # Categorize by variant (cpu, cu12, etc.)
    variants = defaultdict(list)
    for tag, wheel in all_wheels:
        variant = wheel_variant(wheel)
        if variant:
            variants[variant].append((tag, wheel))

    preferred_order = ["cpu", "cu12", "cu13", "rocm6"]
    ordered_variants = [v for v in preferred_order if v in variants] + \
                       sorted(v for v in variants if v not in preferred_order)

    for variant in ordered_variants:
        variant_wheels = variants[variant]

        out_dir = os.path.join("whl_out", variant, package_name)
        os.makedirs(out_dir, exist_ok=True)

        html = generate_index_html(package_name, variant_wheels, repo_slug)
        with open(os.path.join(out_dir, "index.html"), "w") as f:
            f.write(html)

        unique_count = len(set(w for _, w in variant_wheels))
        print(f"Generated index for {variant} with {unique_count} unique wheels.")
