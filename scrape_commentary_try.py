"""
World Cup 2026 Commentary Scraper - v3
Fixes:
  - Scrolls UP to load older commentary entries (LiveScore is newest-first)
  - Clicks "Load more" if present to get full match
  - Fresh driver per match to avoid session timeout crashes
"""

import time
import json
import csv
import os
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

OUTPUT_DIR = "output"
CSV_PATH = "match_urls copy.csv"
WAIT_BETWEEN_MATCHES = 10
SCROLL_PAUSE = 1.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,2000")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    return driver


def safe_filename(name):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def dismiss_overlays(driver):
    """Try to dismiss cookie/consent banners that block clicks."""
    for selector in [
        "[class*='cookie'] button", "[class*='consent'] button",
        "[id*='cookie'] button", "[id*='consent'] button",
        "button[class*='accept']", "button[class*='Accept']",
    ]:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, selector)
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.5)
        except Exception:
            pass


def click_commentary_tab(driver, match_name):
    """Find and click the Commentary tab. Returns False if not found."""
    try:
        tab = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, "//*[contains(text(), 'Commentary')]")
            )
        )
        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center'}); arguments[0].click();",
            tab,
        )
        print(f"  [tab] clicked: {tab.text!r} ({tab.tag_name})")
        time.sleep(3)
        return True
    except TimeoutException:
        print(f"  [!] Commentary tab not found for {match_name}")
        return False


def load_all_commentary(driver):
    """
    LiveScore shows commentary newest-first.
    Older entries are above — scroll UP repeatedly to load them.
    Also click any 'Load more' / 'Show more' buttons that appear.
    """
    # First go to top
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(2)

    # Click "More commentary" repeatedly until it disappears
    # Each click loads one older batch — a full 90-min match may need 5-15 clicks
    for attempt in range(50):  # up to 50 batches, safety cap
        clicked_any = False
        for btn_text in ["More commentary", "Load more", "Show more", "More", "Earlier"]:
            try:
                btn = driver.find_element(
                    By.XPATH, f"//*[contains(text(), '{btn_text}')]"
                )
                driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center'}); arguments[0].click();",
                    btn
                )
                time.sleep(2)  # wait for new batch to render
                clicked_any = True
                print(f"  [load] clicked '{btn_text}' (batch {attempt+1})")
                break  # restart outer loop to re-find button after DOM update
            except NoSuchElementException:
                pass
        if not clicked_any:
            print(f"  [load] no more 'load' buttons found after {attempt} batches")
            break

    # Now scroll down fully to trigger any remaining lazy loads
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(40):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # Scroll back to top one more time and check again
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)


def extract_entries(driver):
    """
    Extract structured commentary using LiveScore's ID-based naming:
      div[id^="match-detail_comment_N"] (parent)
        span[id="match-detail_comment_time_N"] → minute
        div[id="match-detail_comment_text_N"] → text
    """
    comment_divs = driver.find_elements(
        By.CSS_SELECTOR,
        'div[id^="match-detail_comment_"]:not([id*="_time_"]):not([id*="_text_"])'
    )

    entries = []
    for el in comment_divs:
        el_id = el.get_attribute("id")
        idx = el_id.split("_")[-1]
        minute, text = "", ""
        try:
            minute = el.find_element(
                By.CSS_SELECTOR, f'span[id="match-detail_comment_time_{idx}"]'
            ).text.strip()
        except NoSuchElementException:
            pass
        try:
            text = el.find_element(
                By.CSS_SELECTOR, f'div[id="match-detail_comment_text_{idx}"]'
            ).text.strip()
        except NoSuchElementException:
            text = el.text.strip()

        if text:
            entries.append({"minute": minute, "text": text})

    # Deduplicate preserving order
    seen, unique = set(), []
    for e in entries:
        if e["text"] not in seen:
            seen.add(e["text"])
            unique.append(e)

    return unique


def scrape_match(match_name, url):
    """Fresh driver per match — avoids session crashes between matches."""
    print(f"\n--- Scraping: {match_name} ---")
    driver = get_driver()
    try:
        driver.get(url)
        time.sleep(3)
        dismiss_overlays(driver)

        if not click_commentary_tab(driver, match_name):
            return None

        load_all_commentary(driver)
        entries = extract_entries(driver)
        print(f"  -> Found {len(entries)} commentary entries")

        result = {
            "match_name": match_name,
            "url": url,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_entries": len(entries),
            "raw_entries": entries,
        }

        out_path = os.path.join(OUTPUT_DIR, f"{safe_filename(match_name)}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  -> Saved to {out_path}")
        return result

    except Exception as e:
        print(f"  [ERROR] {match_name}: {e}")
        return None
    finally:
        driver.quit()  # Always close, even on crash


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Create {CSV_PATH} first with columns: match_name,url")
        return

    matches = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("match_name") and row.get("url"):
                matches.append((row["match_name"].strip(), row["url"].strip()))

    print(f"Loaded {len(matches)} matches from {CSV_PATH}")

    results = []
    for i, (match_name, url) in enumerate(matches, 1):
        print(f"\n[{i}/{len(matches)}]")
        res = scrape_match(match_name, url)
        results.append(res)
        if i < len(matches):
            print(f"  Waiting {WAIT_BETWEEN_MATCHES}s before next match...")
            time.sleep(WAIT_BETWEEN_MATCHES)

    success = sum(1 for r in results if r)
    print(f"\n{'='*40}")
    print(f"Done. {success}/{len(matches)} matches scraped successfully.")
    print(f"Check the '{OUTPUT_DIR}/' folder for JSON output.")


if __name__ == "__main__":
    main()