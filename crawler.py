from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep

def crawler(user_id, password):
        
    options = webdriver.ChromeOptions()
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")

    driver = webdriver.Chrome(executable_path=r'./chromedriver')

    driver.get('https://everytime.kr/login')
    driver.implicitly_wait(5)

    driver.find_element(By.NAME, 'user_id').send_keys(user_id) # 아이디
    driver.find_element(By.NAME, 'password').send_keys(password) # 비번

    driver.find_element(By.XPATH, '//*[@id="container"]/form/p[3]/input').click()

    driver.get('https://everytime.kr/timetable')


    # 수업 목록 - 검색 클릭
    driver.find_element(By.XPATH, '//*[@id="container"]/ul/li[1]').click()
    # 튀어나온 창 닫기
    driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/a[1]').click()

    sleep(2)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    #'#container > div > div.tablebody > table > tbody > tr > td:nth-child(' + 과목수 + ') > div.cols > div.subject.color' + 몇번째 식으로 접근?
    subs = soup.select('#container > div > div.tablebody > table > tbody > tr')

    results = []
    result = []

    for sub in subs:
        subName = sub.select('h3')
        subProf = sub.select('em')
        subLecRoom = sub.select('span')

        print(subName)
        print(subProf)
        print(subLecRoom)
        result.append(subName, subProf, subLecRoom)
        results.append(result)
        

    return results
    
if __name__=='__main__':
    crawler("aaa", "bbb")