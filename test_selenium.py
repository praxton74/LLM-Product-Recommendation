from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])  # Suppresses certain Chrome logs

driver = webdriver.Chrome(options=chrome_options)

driver.get("http://127.0.0.1:8000")

question_input = driver.find_element(By.NAME, "question")
submit_button = driver.find_element(By.XPATH, "//button[@type='submit']")

question_input.send_keys("What is AI in one sentence?")
submit_button.click()

# Wait for response 
time.sleep(3)

# Assertions and further actions
# ...

# Close the WebDriver
driver.quit()
