import sys
from typing import TextIO


class CustomStream(object):
    """自定义流, 将标准输入输出流进行重新封装, 对输入输出功能进行扩充

    Attribute:
        stream (TextIO): 输入输出流
        log (TextIOWrapper): 日志文件对象  # 新增
        message (str): 读取的文本信息

    """
    stream: TextIO
    log: TextIO
    message: str

    def __init__(self, filename, stream: TextIO):  # 新增 filename 参数
        self.stream = stream
        self.log = open(filename, 'w', encoding="UTF-8")  # 新增

    def write(self, message):
        self.stream.write(message)
        self.log.write(message)  # 新增
        self.log.flush()  # 新增

    def readline(self):
        message = self.stream.readline()
        self.log.write(message)  # 新增
        self.log.flush()  # 新增
        return message

    def flush(self):
        self.stream.flush()


def terminal2file_on(stdout_filename='x.log', stderr_filename='x.err'):
    sys.stdout = CustomStream(stdout_filename, sys.stdout)
    sys.stderr = CustomStream(stderr_filename, sys.stderr)


''' usage
sys.path.append('../..')
from tools import terminal2file
'''