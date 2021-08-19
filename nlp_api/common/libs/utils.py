

from flask.signals import message_flashed
from pypinyin import lazy_pinyin
import random 
import xml.etree.ElementTree as et


import string
import time 
from flask import make_response
import hashlib
import xmltodict
import requests


class Resp:
    def __init__(code = 200, msg = "success", data = []):
        pass 


def serialize(o):
    o = o.__dict__
    new_dict = {}
    for k, v in o.items():
        if type(v) is str or type(v) is int or type(v) is float:
            new_dict[k] = v 
        elif str(k)[0] != "_":
            new_dict[k] = str(v)
    return new_dict

def result_to_dict(m):
    d, a = {}, []
    ## 把执行sql语句返回的结果转为dict
    for rowproxy in m:
        # rowproxy.items() returns an array like [(key0, value0), (key1, value1)]
        for column, value in rowproxy.items():
            # build up the dictionary
            d = {**d, **{column: value}}
        a.append(d)
    return a

def verify_prov(prov):
    if prov == "" or prov not in ["北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
                                  "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
                                  "湖北", "湖南", "广东", "广西", "海南","重庆", "四川", "贵州",
                                  "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "香港", "澳门", "台湾"]:
        return ""
    prov_pinyin = lazy_pinyin(prov)
    p = ""
    for each in prov_pinyin:
        p += each
    return p

def justi_prov(prov):
    if prov == "" or prov not in ["北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
                                  "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
                                  "湖北", "湖南", "广东", "广西", "海南","重庆", "四川", "贵州",
                                  "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "香港", "澳门", "台湾"]:
        return False
    else:
        return True

def random_str(length):
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, length))
    return ran_str

def md5(string_request):
    
    # 创建md5对象
    hl = hashlib.md5()

    hl.update(string_request.encode(encoding='utf-8'))

    # print('MD5加密前为 ：' + string_request)
    # print('MD5加密后为 ：' + hl.hexdigest())
    res = hl.hexdigest()
    # print(res)

    return res 

def new_order_id():
    ##新建一个订单号
    now_time = time.time()
    return str(now_time).replace(".", "_") + random_str(6)

class WX_PayToolUtil():
    def __init__(self, APP_ID, MCH_ID, API_KEY):
        self._APP_ID = APP_ID # 小程序ID
        self._MCH_ID = MCH_ID # # 商户号
        self._API_KEY = API_KEY
        self._UFDODER_URL = "https://api.mch.weixin.qq.com/pay/unifiedorder" # 接口链接
        self._NOTIFY_URL = "https://www.shengbaiedu.com/pay_back" # 异步通知
        self.SERVER_IP = "8.140.177.178"
  
    def generate_sign(self, param):
        stringA = ''
        ks = sorted(param.keys())
        # 参数排序
        for k in ks:
            stringA += (k + '=' + param[k] + '&')
        # 拼接商户KEY
        stringSignTemp = stringA + "key=" + self._API_KEY ### key还没有
        # md5加密,也可以用其他方式
        hash_md5 = hashlib.md5(stringSignTemp.encode('utf8'))
        sign = hash_md5.hexdigest().upper()
        return sign

    def getPayUrl(self, orderid, openid, goodsPrice, body):
        """向微信支付端发出请求，获取url"""
        key = self._API_KEY
        nonce_str = random_str(30) # 生成随机字符串，小于32位
        params = {
        'appid': self._APP_ID, # 小程序ID
        'mch_id': self._MCH_ID, # 商户号
        'nonce_str': nonce_str, # 随机字符串
        "body": body, # 支付说明
        'out_trade_no': orderid, # 生成的订单号
        'total_fee': str(goodsPrice), # 标价金额
        'spbill_create_ip': "127.0.0.1", # 小程序不能获取客户ip，web用socekt实现
        'notify_url': self._NOTIFY_URL,
        'trade_type': "JSAPI", # 支付类型
        "openid": openid, # 用户id
        }
        # 生成签名
        params['sign'] = self.generate_sign(params)
        # python3一种写法
        param = {'root': params}
        xml = xmltodict.unparse(param)
        response = requests.post(self._UFDODER_URL, data=xml.encode('utf-8'), headers={'Content-Type': 'text/xml'})
        # xml 2 dict
        response.encoding = response.apparent_encoding
        msg = response.text
        # print(msg)
        xmlmsg = xmltodict.parse(msg)
        # 4. 获取prepay_id
        # print(xmlmsg)
        # print(xmlmsg['xml']['return_msg'].encode("utf-8"))
        if xmlmsg['xml']['return_code'] == 'SUCCESS':
            if xmlmsg['xml']['result_code'] == 'SUCCESS':
                prepay_id = xmlmsg['xml']['prepay_id']
                # 时间戳
                timeStamp = str(int(time.time()))
                # 5. 五个参数
                data = {
                "appId": self._APP_ID,
                "nonceStr": nonce_str,
                "package": "prepay_id=" + prepay_id,
                "signType": 'MD5',
                "timeStamp": timeStamp,
                }
                # 6. paySign签名
                paySign = self.generate_sign(data)
                data["paySign"] = paySign # 加入签名
                # 7. 传给前端的签名后的参数
                return data
        else:
            data = {}
            data["msg"] = xmlmsg['xml']['return_msg']
            return data 

def payback(request):
    try:
        data = request.get_data()
        data = data.decode('utf-8')
        root = et.fromstring(data)
        print(root)
        return_code = root.find('.//return_code')
        out_trade_no = root.find('.//out_trade_no')
        # msg = request.body.decode("utf-8")
        # print(msg)
        # xmlmsg = xmltodict.parse(msg)
        # print(xmlmsg)
        # return_code = xmlmsg['xml']['return_code']
        if return_code == 'FAIL':
        # 官方发出错误
            response = make_response("""<xml><return_code><![CDATA[FAIL]]></return_code>
                <return_msg><![CDATA[Signature_Error]]></return_msg></xml>""", 200)
            response.headers['Content-Type'] = 'text/xml'

            return response
        elif return_code == 'SUCCESS':
        # 拿到这次支付的订单号
            # out_trade_no = xmlmsg['xml']['out_trade_no']
            # 根据需要处理业务逻辑
            response = make_response("""<xml><return_code><![CDATA[SUCCESS]]></return_code>
                <return_msg><![CDATA[OK]]></return_msg></xml>""", 200)
            response.headers['Content-Type'] = 'text/xml'
            return response
    except Exception as e : 
        print(e)
        response = make_response("""<xml><return_code><![CDATA[FAIL]]></return_code>
                <return_msg><![CDATA[Signature_Error]]></return_msg></xml>""", 200)
        response.headers['Content-Type'] = 'text/xml'

        return response

if __name__ == "__main__":
    # print(random_str(10))

    # md5("dsada")

    # print(new_order_id())

    # print(generate_sign({"appid": "dsdad", "body": "21eds"}))

    pay_tool = WX_PayToolUtil("appid", "mchid", "apikey", "notify_url")
    order_id = new_order_id()
    data = pay_tool.getPayUrl(orderid=order_id, openid="sdsd", goodsPrice="21")

    print(data)