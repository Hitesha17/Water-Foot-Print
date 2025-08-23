from icrawler.builtin import GoogleImageCrawler

def download_images(query, folder_name, max_images=200):
    crawler = GoogleImageCrawler(storage={"root_dir": f"data/{folder_name}"})
    crawler.crawl(keyword=query, max_num=max_images)

# Refined search queries
categories = [
    "blue water crops",
    "green water crops",
    "grey water crops",
    "crops with blue water",
    "crops with green water",
    "crops with grey water"
]

for category in categories:
    download_images(category, category.replace(" ", "_"), max_images=200)
