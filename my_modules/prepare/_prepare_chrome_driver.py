from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def prepare_chrome_driver():

    options = Options()
    options.add_argument('--headless')
    options.add_argument("--no-sandbox")
    # selenium4 を使用せよ
    driver = webdriver.Chrome( options=options)
    driver.set_window_size(50, 50)
    return driver