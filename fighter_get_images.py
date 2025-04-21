from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import csv

# Set up Chrome options with the correct binary location
chrome_options = Options()
chrome_options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"  # Adjust if needed

# Set up Selenium WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

base_url = "https://www.ufc.com/athletes/all"
athlete_data = []
page = 1

while True:
    # Construct the URL for the current page
    url = f"{base_url}?page={page}" if page > 1 else base_url
    driver.get(url)

    # Parse the page
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    flipcards = soup.find_all('div', class_='c-listing-athlete-flipcard')

    # Extract athlete names and image URLs from the current page
    for card in flipcards:
        # Get the athlete name
        name_tag = card.find('span', class_='c-listing-athlete__name')
        name = name_tag.get_text(strip=True) if name_tag else "Unknown Athlete"

        # Get the image URL from the front of the flipcard
        front_div = card.find('div', class_='c-listing-athlete-flipcard__front')
        img_tag = front_div.find('img') if front_div else None
        image_url = img_tag['src'] if img_tag and 'src' in img_tag.attrs else "No Image"

        # Store the name and image URL as a tuple
        athlete_data.append((name, image_url))

    # Check if there's a "Load More" link to determine if more pages exist
    load_more_link = soup.find('a', class_='button', rel='next')
    if not load_more_link:
        print(f"No more pages to load. Stopped at page {page}.")
        break

    page += 1

# Save the data to a CSV file
with open('Datasets/athlete_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Athlete Name', 'Image URL'])  # Write header
    writer.writerows(athlete_data)  # Write data rows

print(f"Scraped {len(athlete_data)} athletes and saved to athlete_data.csv")
driver.quit()