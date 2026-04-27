"""
Hard-negative mining from TheRealReal (Playwright edition)
==========================================================

TheRealReal professionally authenticates every Hermès bag they sell, so any
listing image they publish should ideally be predicted as `Real` by the
Hermès Authenticator model. Anything the model predicts as `Fake` here is a
false positive — a hard negative we want in the training set.

This script uses **Playwright** (a real Chromium) to defeat the HTTP-403
anti-bot response, then reuses the browser session to download images
without tripping the same detector. Inference is posted to a locally
running FastAPI server (http://localhost:8000).

Prereqs
-------
    pip install playwright requests
    playwright install chromium

Usage
-----
    python mine_hard_negatives.py                    # 50 listings, t=0.70
    python mine_hard_negatives.py --listings 100 --threshold 0.85
    python mine_hard_negatives.py --out Dataset/HardNegatives
    python mine_hard_negatives.py --dry-run          # run inference, don't save
    python mine_hard_negatives.py --headless         # run without showing browser

Runs HEADFUL by default. On the first run TheRealReal shows a "Press & Hold"
bot challenge — hold it and the script waits until the product grid renders,
then proceeds. Cookies persist in `.mining_browser_profile/`, so subsequent
runs go straight to scraping.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    sys.exit("Missing dependency: pip install playwright && playwright install chromium")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CATEGORY_URL = "https://www.therealreal.com/products?keywords=birkin"
PAGE_URL_TMPL = "https://www.therealreal.com/products?keywords=birkin&page={page}"

REAL_CHROME_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Playwright-based image-URL collection
# ---------------------------------------------------------------------------

def _pick_largest_srcset(srcset: str) -> str | None:
    """Given an srcset attr value, return the largest-width candidate URL."""
    if not srcset:
        return None
    best_url, best_w = None, -1
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split()
        url = bits[0]
        w = 0
        if len(bits) > 1 and bits[1].endswith("w"):
            try:
                w = int(bits[1][:-1])
            except ValueError:
                w = 0
        if w > best_w:
            best_w, best_url = w, url
    return best_url


# JS pulled out for clarity. Runs inside the page and returns a list of
# best-quality product-image URLs. Strategy:
#   1. Walk every <img> on the page (not just those inside `a[href*="/products/"]`
#      anchors — TheRealReal's card markup doesn't always wrap images that way).
#   2. Keep images whose best candidate URL is on a TheRealReal CDN OR lives
#      inside an anchor that links to a product page.
#   3. Drop obvious UI chrome (logo/sprite/banner/avatar/etc.) and small icons.
PAGE_EXTRACT_JS = r"""
() => {
  const out = [];
  const seen = new Set();
  const bad = ['logo','sprite','icon-','banner','placeholder','avatar',
               'favicon','flag','chevron','/icons/','/ui/','spinner',
               '/prismic/','plpcard','hero-','editorial','promo-'];

  document.querySelectorAll('img').forEach(img => {
    const candidates = [];
    const src = img.currentSrc || img.src;
    if (src) candidates.push([src, img.naturalWidth || 0]);
    const ss = img.getAttribute('srcset') || img.getAttribute('data-srcset');
    if (ss) {
      ss.split(',').forEach(p => {
        const [u, w] = p.trim().split(/\s+/);
        if (u) candidates.push([u, parseInt((w||'0').replace('w',''))||0]);
      });
    }
    ['data-src','data-lazy-src','data-original'].forEach(attr => {
      const v = img.getAttribute(attr);
      if (v) candidates.push([v, 0]);
    });
    candidates.sort((a,b) => b[1]-a[1]);
    if (!candidates.length) return;

    let url = candidates[0][0];
    if (url.startsWith('//')) url = 'https:' + url;
    if (!/^https?:\/\//i.test(url)) return;
    if (!/\.(jpg|jpeg|png|webp)(\?|$)/i.test(url)) return;
    if (bad.some(b => url.toLowerCase().includes(b))) return;

    // Size check: exclude tiny icons. When dimensions are known (loaded),
    // require >=150px. If image hasn't loaded yet, let it through.
    const nw = img.naturalWidth || 0;
    const nh = img.naturalHeight || 0;
    if (nw > 0 && nw < 150) return;
    if (nh > 0 && nh < 150) return;

    // Signal #1: lives inside an anchor that links to /products/...
    const anchor = img.closest('a[href]');
    const onProductAnchor = !!(anchor && /\/products\//.test(anchor.getAttribute('href') || ''));

    // Signal #2: the image URL itself is on a TheRealReal CDN host
    let host = '';
    try { host = new URL(url).hostname.toLowerCase(); } catch (e) {}
    const onProductCDN = host.includes('therealreal');

    if (!onProductAnchor && !onProductCDN) return;

    if (seen.has(url)) return;
    seen.add(url);
    out.push(url);
  });

  return out;
}
"""


# Runs inside the page and returns diagnostic counts so we can tell whether
# the selector is wrong or the page never rendered.
PAGE_DEBUG_JS = r"""
() => {
  const imgs = Array.from(document.querySelectorAll('img'));
  const prodAnchors = Array.from(document.querySelectorAll('a[href*="/products/"]'));
  const botFlags = {
    pxCaptcha: !!document.querySelector('#px-captcha'),
    captchaIframe: !!document.querySelector('iframe[src*="captcha"], iframe[id*="px-"]'),
    pressHoldText: /press.{0,3}(and|&).{0,3}hold/i.test((document.body && document.body.innerText) || ''),
  };
  return {
    url: location.href,
    title: document.title,
    totalImgs: imgs.length,
    imgsWithSrcset: imgs.filter(i => i.getAttribute('srcset') || i.getAttribute('data-srcset')).length,
    productAnchors: prodAnchors.length,
    sampleAnchorHrefs: prodAnchors.slice(0,5).map(a => a.getAttribute('href')),
    sampleImgSrcs: imgs.slice(0,10).map(i => i.currentSrc || i.src || i.getAttribute('data-src') || '').filter(Boolean),
    botFlags,
  };
}
"""


# Returns true if TheRealReal's PerimeterX press-and-hold is currently showing.
BOT_DETECT_JS = r"""
() => {
  if (document.querySelector('#px-captcha')) return true;
  if (document.querySelector('iframe[src*="captcha"]')) return true;
  if (document.querySelector('iframe[id*="px-"]')) return true;
  const txt = (document.body && document.body.innerText) || '';
  if (/press.{0,3}(and|&).{0,3}hold/i.test(txt)) return true;
  return false;
}
"""


def wait_for_bot_challenge_clear(page, max_wait: float = 120.0, verbose: bool = True) -> bool:
    """
    Poll the page — if a press-and-hold / PerimeterX challenge is active, wait
    (up to `max_wait`s) until the user clears it. Returns True if the page is
    clean when we return, False if we timed out.
    """
    deadline = time.time() + max_wait
    announced = False
    while time.time() < deadline:
        try:
            active = page.evaluate(BOT_DETECT_JS)
        except Exception:
            active = False
        if not active:
            if announced and verbose:
                print("  [info] bot challenge cleared — continuing.")
            return True
        if not announced and verbose:
            print(f"  [info] bot challenge detected on this page. "
                  f"Please solve it (press & hold) — waiting up to {int(max_wait)}s.")
            announced = True
        time.sleep(1.5)
    if verbose:
        print(f"  [warn] bot challenge still present after {int(max_wait)}s "
              "— continuing anyway, extraction may yield 0 images.")
    return False


def scroll_to_bottom(page, step: int = 900, pause: float = 0.6, max_rounds: int = 25):
    """Scroll in chunks so lazy-loaded images actually fetch."""
    last_h = 0
    for _ in range(max_rounds):
        h = page.evaluate("document.documentElement.scrollHeight")
        if h == last_h:
            break
        last_h = h
        for y in range(0, h, step):
            page.evaluate(f"window.scrollTo(0, {y})")
            time.sleep(0.05)
        page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        time.sleep(pause)


def collect_image_urls(page,
                       target_new: int,
                       already_seen: set[str],
                       start_page: int,
                       max_pages: int,
                       delay: float,
                       debug: bool = False,
                       bot_wait: float = 120.0,
                       verbose: bool = True) -> list[str]:
    """
    Walk category pages and collect up to `target_new` product images that are
    NOT already in `already_seen`. Keeps paging until we have enough, or hit
    `max_pages`, or see a page with no new images.

    On EVERY page we check for TheRealReal's press-and-hold bot challenge and
    wait (up to `bot_wait`s) until the user clears it before scraping.
    """
    fresh: list[str] = []
    seen_this_run: set[str] = set()

    page_num = start_page
    empty_pages_in_a_row = 0
    while len(fresh) < target_new and page_num <= max_pages:
        url = PAGE_URL_TMPL.format(page=page_num)
        if verbose:
            print(f"[crawl] page {page_num} -> {url}")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        except PWTimeout:
            print(f"  [warn] navigation timeout on page {page_num}")
            break

        # Short settle before checking for a bot challenge. PerimeterX often
        # takes ~1–2s to inject its iframe after DOMContentLoaded.
        time.sleep(2)

        # Detect + wait for press-and-hold on EVERY page. TheRealReal can
        # re-challenge mid-session, especially when paging fast.
        wait_for_bot_challenge_clear(page, max_wait=bot_wait, verbose=verbose)

        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except PWTimeout:
            pass
        scroll_to_bottom(page)
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except PWTimeout:
            pass

        # Second check: scrolling/rendering sometimes triggers a new challenge.
        if page.evaluate(BOT_DETECT_JS):
            wait_for_bot_challenge_clear(page, max_wait=bot_wait, verbose=verbose)
            scroll_to_bottom(page)

        if debug:
            try:
                info = page.evaluate(PAGE_DEBUG_JS)
                print(f"  [debug] url      : {info.get('url')}")
                print(f"  [debug] title    : {info.get('title')}")
                print(f"  [debug] <img>    : {info.get('totalImgs')} total, "
                      f"{info.get('imgsWithSrcset')} with srcset")
                print(f"  [debug] product anchors: {info.get('productAnchors')}")
                hrefs = info.get('sampleAnchorHrefs') or []
                for h in hrefs[:3]:
                    print(f"  [debug]   href: {h}")
                srcs = info.get('sampleImgSrcs') or []
                for s in srcs[:5]:
                    print(f"  [debug]   img : {s[:120]}")
                flags = info.get('botFlags') or {}
                if any(flags.values()):
                    print(f"  [debug] BOT FLAGS: {flags}")
            except Exception as e:
                print(f"  [debug] could not read debug info: {e}")

        urls = page.evaluate(PAGE_EXTRACT_JS)
        added_this_page = 0
        skipped_already_seen = 0
        for u in urls:
            if u.startswith("//"):
                u = "https:" + u
            if not u.startswith("http"):
                continue
            if u in seen_this_run:
                continue
            seen_this_run.add(u)
            if u in already_seen:
                skipped_already_seen += 1
                continue
            fresh.append(u)
            added_this_page += 1
            if len(fresh) >= target_new:
                break

        if verbose:
            print(f"  [info] page {page_num}: +{added_this_page} new "
                  f"(skipped {skipped_already_seen} already-seen, "
                  f"fresh total: {len(fresh)})")

        if added_this_page == 0 and skipped_already_seen == 0:
            empty_pages_in_a_row += 1
            if empty_pages_in_a_row >= 2:
                print("  [info] two empty pages in a row — end of results.")
                break
        else:
            empty_pages_in_a_row = 0

        page_num += 1
        time.sleep(delay)

    return fresh[:target_new]


def load_seen_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_seen_url(path: Path, url: str) -> None:
    with path.open("a") as f:
        f.write(url + "\n")


def download_via_browser(context, url: str, timeout_ms: int = 20_000) -> bytes | None:
    """Fetch an image reusing the browser's cookies / referrer to avoid 403s."""
    try:
        r = context.request.get(url, timeout=timeout_ms)
    except Exception as e:
        print(f"  [warn] failed to download {url}: {e}")
        return None
    if r.status != 200:
        print(f"  [warn] HTTP {r.status} downloading {url}")
        return None
    return r.body()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(api_base: str, image_bytes: bytes, filename: str) -> dict | None:
    url = api_base.rstrip("/") + "/api/predict?gradcam=false"
    try:
        r = requests.post(
            url,
            files={"file": (filename, image_bytes, "application/octet-stream")},
            timeout=60.0,
        )
    except requests.RequestException as e:
        print(f"  [warn] inference error: {e}")
        return None
    if r.status_code != 200:
        print(f"  [warn] API HTTP {r.status_code}: {r.text[:200]}")
        return None
    return r.json()


def safe_filename_from_url(url: str, content: bytes) -> str:
    parsed = urlparse(url)
    stem = Path(parsed.path).stem or "image"
    stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)[:60]
    digest = hashlib.sha1(content).hexdigest()[:10]
    ext = Path(parsed.path).suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    return f"realreal_{stem}_{digest}{ext}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listings", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--out", default="Dataset/Real")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="Seconds between page navigations / inference calls.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--headless", action="store_true",
                    help="Run Chromium in headless mode. Default is headful because "
                         "TheRealReal's bot challenge needs manual solving once.")
    ap.add_argument("--profile",
                    default=".mining_browser_profile",
                    help="Directory for the persistent browser profile. Keeps "
                         "cookies between runs so the bot challenge is one-and-done.")
    ap.add_argument("--start-page", type=int, default=1,
                    help="Category page to start on (default 1).")
    ap.add_argument("--max-pages", type=int, default=40,
                    help="Stop after this many pages even if target not reached.")
    ap.add_argument("--seen-log", default=".mining_seen_urls.txt",
                    help="File that tracks URLs screened in previous runs. "
                         "Fresh runs skip URLs already listed here.")
    ap.add_argument("--reset-seen", action="store_true",
                    help="Clear the seen-URL log before starting.")
    ap.add_argument("--debug", action="store_true",
                    help="Print DOM diagnostics on every page (image/anchor "
                         "counts, sample URLs, bot-challenge flags).")
    ap.add_argument("--bot-wait", type=float, default=120.0,
                    help="Max seconds to wait on each page if a press-and-hold "
                         "bot challenge appears (default 120).")
    args = ap.parse_args()

    root = Path(__file__).parent.resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_log_path = (root / args.seen_log).resolve()
    if args.reset_seen and seen_log_path.exists():
        seen_log_path.unlink()
        print(f"[info] cleared {seen_log_path.name}")
    already_seen = load_seen_urls(seen_log_path)

    print("=" * 72)
    print("Hermès Authenticator — hard negative mining (Playwright)")
    print("=" * 72)
    print(f"Target NEW      : {args.listings}")
    print(f"Fake threshold  : {args.threshold:.2f}")
    print(f"Save directory  : {out_dir}")
    print(f"API endpoint    : {args.api}")
    print(f"Headless        : {args.headless}")
    print(f"Browser profile : {args.profile}")
    print(f"Start page      : {args.start_page}  (max {args.max_pages})")
    print(f"Seen URL log    : {seen_log_path.name}  ({len(already_seen)} already logged)")
    print(f"Bot-wait        : {args.bot_wait:.0f}s per page")
    print(f"Debug DOM       : {args.debug}")
    print(f"Dry run         : {args.dry_run}")
    print()

    # Pre-flight: server reachable?
    try:
        r = requests.get(args.api.rstrip("/") + "/api/health", timeout=5)
        r.raise_for_status()
        health = r.json()
        if not health.get("model_path"):
            print("[fatal] Server up but model file not found.")
            return 2
        print(f"[ok] Server online · device={health.get('device')} · "
              f"classes={health.get('classes')}")
    except Exception as e:
        print(f"[fatal] Could not reach server at {args.api}: {e}")
        print("       Start it with:  uvicorn app:app --reload --port 8000")
        return 2
    print()

    stats = {"total": 0, "real": 0, "fake": 0, "saved": 0, "errors": 0}

    # Persistent browser profile: cookies and the bot-challenge solution
    # survive across runs, so users only have to press-and-hold once.
    profile_dir = (root / args.profile).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=args.headless,
            user_agent=REAL_CHROME_UA,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
        )
        # Small stealth touch: hide webdriver flag that trivial detectors check
        context.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        )
        page = context.pages[0] if context.pages else context.new_page()

        # Warm-up: visit the homepage first so cookies get set naturally
        try:
            page.goto("https://www.therealreal.com/", wait_until="domcontentloaded", timeout=30_000)
            time.sleep(1.5)
        except PWTimeout:
            pass

        print(f"[step 1] Collecting up to {args.listings} NEW images from TheRealReal…")
        image_urls = collect_image_urls(
            page,
            target_new=args.listings,
            already_seen=already_seen,
            start_page=args.start_page,
            max_pages=args.max_pages,
            delay=args.delay,
            debug=args.debug,
            bot_wait=args.bot_wait,
        )
        if not image_urls:
            print("[fatal] No images collected. The page structure may have changed, "
                  "or the site is still blocking.")
            context.close()
            return 3
        print(f"[ok] Collected {len(image_urls)} candidate images.")
        print()

        print("[step 2] Running inference on each image…")
        for i, url in enumerate(image_urls, 1):
            print(f"[{i:>3}/{len(image_urls)}] {url[:90]}")
            content = download_via_browser(context, url)
            if not content:
                stats["errors"] += 1
                append_seen_url(seen_log_path, url)  # don't retry next time
                continue
            filename = safe_filename_from_url(url, content)

            # Hash-based dedupe against what's already on disk:
            dest = out_dir / filename
            if dest.exists():
                print(f"        (already saved — {filename}, skipping)")
                append_seen_url(seen_log_path, url)
                continue

            result = run_inference(args.api, content, filename)
            if not result:
                stats["errors"] += 1
                append_seen_url(seen_log_path, url)
                continue
            stats["total"] += 1

            verdict = result.get("verdict", "?")
            confidence = float(result.get("confidence", 0.0))
            probs = result.get("probs", {})
            p_fake = float(probs.get("Fake", 0.0))

            if verdict.lower() == "real":
                stats["real"] += 1
                print(f"        → REAL  (conf {confidence:.3f})")
            else:
                stats["fake"] += 1
                print(f"        → FAKE  (conf {confidence:.3f}, p_fake={p_fake:.3f})"
                      "  ← potential hard negative")
                if p_fake >= args.threshold and not args.dry_run:
                    dest.write_bytes(content)
                    stats["saved"] += 1
                    print(f"        ✓ saved to {dest.relative_to(root)}")

            append_seen_url(seen_log_path, url)
            time.sleep(args.delay * 0.4)

        context.close()

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Images screened      : {stats['total']}")
    print(f"    → predicted Real   : {stats['real']}")
    fp_rate = (stats["fake"] / stats["total"] * 100) if stats["total"] else 0.0
    print(f"    → predicted Fake   : {stats['fake']}  ({fp_rate:.1f}% FPR on TheRealReal)")
    print(f"  Errors               : {stats['errors']}")
    print(f"  Hard negatives saved : {stats['saved']}  (threshold ≥ {args.threshold:.2f})")
    if args.dry_run:
        print("  (dry-run: nothing was written to disk)")
    print(f"  Destination          : {out_dir}")
    print()
    print("Next step: retrain — the new samples are already in the Real class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
