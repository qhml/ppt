"""
    @Time    : 03/04/2023
    @Author  : qinghua
    @Software: PyCharm
    @File    : utils.py

"""
from datetime import datetime


def get_current_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
