#!/usr/bin/env python3
"""Checks every rule of Training/cleaning.py: what must go, and typed text that must stay.

    python3 Training/test_cleaning.py
"""
import sys

from cleaning import clean, redact

REDACT = [
    # (text, expected after redact)
    ("邮箱abc.d@qq.com找我", "邮箱＊找我"),
    ("网店www.hongkongdg.com开张", "网店＊开张"),
    ("网址http://a.cn/x好", "网址＊好"),
    ("看 www.abc.com 吧", "看 ＊ 吧"),
    ("官方网站(yys.163.com)、公众号", "官方网站(＊)、公众号"),
    ("e.google.cn 上的 JavaScript", "＊ 上的 JavaScript"),
    ("占豪（微信号：caffe007）看法是", "占豪（微信号：＊）看法是"),
    ("小编微信号:wzadcm5【长按复制】", "小编微信号:＊【长按复制】"),
    ("加我QQ 12345吧", "加我QQ ＊吧"),
    ("作者姓名 李建国 地址", "作者姓名 ＊ 地址"),
    ("联系人：侯经理", "联系人：＊"),
    ("回复 @联合光子a 你说得对", "回复 @＊ 你说得对"),
    ("群里@所有人今天下午三点开会请大家准时到场", "群里@＊"),
    ("双井百环家园南门11楼1211 提前电联", "双井百环家园南门＊ 提前电联"),
    ("天长路56号第二单元104 邮编", "天长路56号＊ 邮编"),
    ("3号楼2单元401室", "＊"),
    ("灰色豫AG0T02宝马", "灰色＊宝马"),
    ("电话13812345678", "电话＊"),
    ("电话138 1234 5678", "电话＊"),
    ("身份证11010119900307123X", "身份证＊"),
    ("加我 wx abc123 吧", "加我 wx ＊ 吧"),
    ("母婴俱乐部，微信号：muying-club（长按复制）", "母婴俱乐部，微信号：＊（长按复制）"),
    ("据jsonline.com报道，带着35胜", "据＊报道，带着35胜"),
    ("真理在个人微博(http://t.cn/abc上对他进行了讽刺", "真理在个人微博(＊上对他进行了讽刺"),
    ("梦哦 httpwww.bilibili.comvideoav123 韩率", "梦哦 ＊ 韩率"),
    ("链接在这httpmp.weixin.qq.comsbizMzA5NTcz最近", "链接在这＊最近"),
    ("欢迎加小编 V信：wenwan335", "欢迎加小编 V信：＊"),
    ("但4603室的业主不在", "但＊室的业主不在"),
    ("他都要讨六两白酒喝。 7http://www.abc.com/x", "他都要讨六两白酒喝。 7＊"),
    ("雷霆VS国王increasewww.vn1777.com-皇冠", "雷霆VS国王increase＊皇冠"),
    ("拳皇小游戏-7k7k.com单人小游戏", "拳皇小游戏-＊单人小游戏"),
    ("意者来信或line：bear2020详谈", "意者来信或line：＊详谈"),
    ("★联络 skype :hot.9932", "★联络 skype :＊"),
    ("抖音号：abc_12345", "抖音号：＊"),
    ("道他是在这里认识的 mjhkuu.tk/?id=3 问好", "道他是在这里认识的 ＊ 问好"),
    ("网址很好记：now.to/x5c,希望", "网址很好记：＊,希望"),
    ("图片来自：my365.in/hiyok", "图片来自：＊"),
    ("新西兰 ii.nz 的项目经理", "新西兰 ＊ 的项目经理"),
    ("书在 zh.zlibv.ru/book/123 下载", "书在 ＊ 下载"),
    ("还可以使用 crontab.guru 这类网站", "还可以使用 ＊ 这类网站"),
    ("枸杞茶。U04M04XXXTVHF5S74H04F8M45W3PBXT3 用棉布", "枸杞茶。＊ 用棉布"),
    ("记忆力,5U7mn9lidv1wpnPdkpRXocip1J5ekCVyPzjGKncPnnyXw 好", "记忆力,＊ 好"),
]
KEEP = [
    # typed text that must come out unchanged
    "我们2016-07-20 11:30在公司门口见面吧",
    "北京时间2011年1月2日8:30，热火将主场迎战金州勇士队",
    "会议时间2016年10月27日8:30～17:00北京友谊宾馆",
    "价格是1299元，2024年买的",
    "姓名学没什么道理",
    "第一单元 30页的题",
    "为啥你的不显示来自820客户端了",
    "我住8楼1000块一个月",
    "wxWidgets 和 Qt 都可以",
    "Node.js 和 bar.md",
    "管住手！！！！",
    "唉......",
    "+1",
    "版本 1.2.3 修好了",
    "赤鲷の改姓名\n改名换姓做皇民",
    "I'll go on QQ immediately.",
    "user.Info 接口所需的用户信息",
    "http协议和https的区别",
    "办公室在三楼",
    "iPhone15ProMax 很好用",
    "text-decoration-line: solid 替代",
    "这条 line 很长",
    "一个 CanvasRenderingContext2DSettings 对象",
    "比雷诺RenaultMeganeRS275Trophy-R在纽北的成绩",
    "NodeCIDRMaskSizeIPv6 为双栈集群",
    "脚本 update-imported-docs.py 位于",
    "包含文件 index.md 和文章",
    "通过请求参数 req.params.id 在控制器中",
    "设置 memory.oom.group 标志",
    "原先的网址是tianya.at，现在改为tianya.my",
]
CLEAN = [
    # (document, expected after clean)
    ("作者：wen1984年 回复日期：2011-04-23 18:29:52 偶和偶的印度男友", "偶和偶的印度男友"),
    ("@lijianguo056 8楼 2010-01-03 10:15:00\n说得好", "说得好"),
    ("@风过无雨痕 10楼 2015-09-09 08:18:00\n赞同", "赞同"),
    ("回复第7楼(作者:@枫叶轩主人 于 2013-02-17 18:56)\n美狗去死", "美狗去死"),
    ("报 回复 作者：eqikan 来自：Android客户端 时间：2014-05-11 10:00:00", "报 回复"),
    ("大家都是在盼涨[size=14px]，但猪价[/size]能", "大家都是在盼涨，但猪价能"),
    ("每组间隔休息1-2分钟<img data-rawwidth=\"600\" src=\"x.jpg\">题", "每组间隔休息1-2分钟题"),
    ("形状生成器工具可用于定义使用 &lt;basic-shape&gt;", "形状生成器工具可用于定义使用 <basic-shape>"),
    ("黎姿&nbsp;高龄产双胞胎", "黎姿 高龄产双胞胎"),
    ("2019.2.22\n今天天气好", "今天天气好"),
    ("没有知识一定改变不了命运。<img data-rawheight前一段时间", "没有知识一定改变不了命运。前一段时间"),
    ("为什么相对<a class冷冷清清", "为什么相对冷冷清清"),
    ("有人知道吗?\n<a href=\"ima", "有人知道吗?"),
    ("已把你的心门叩开</p\n爱如雪", "已把你的心门叩开\n爱如雪"),
    ("消暑 [/size\n在宜兰", "消暑\n在宜兰"),
    ("M还我锦绣河山M 3楼 2013-06-16 12:25:15\n早上好", "早上好"),
    ("相思苦！赏了。\n27楼 2015-09-10 13:30:00\n好", "相思苦！赏了。\n好"),
    ("搜集 发布者：集优小帝 发表日期：2013年05月21日 正文", "搜集 正文"),
    ("孤独的地球\n【本贴由刘天雨于2011-1-13 22:38:34在〖长安诗歌节〗发表】\n图雅", "孤独的地球\n图雅"),
    ("大屠杀。\n本贴由军火库管理员于2005年7月27日12:43:10在〖军火库〗发表.\n美国", "大屠杀。\n美国"),
    ("。您的群山静泊账号已被涯叔卫士封杀，解封时间为： 2018-08-25 16:24:48 ，在封杀期间不能发表\n好", "好"),
    ("找借口封杀圣僧。\n不染一尘的圣僧： 举报 2017-03-23 12:51:50 评论\n请裁判", "找借口封杀圣僧。\n请裁判"),
    ("发布时间：2016-06-01 12:00:11 编辑：空山鸟语\n几次翘首", "几次翘首"),
    ("第一次被摸胸（转载）(2007-05-25 20:42:52)\n一旦没有了", "第一次被摸胸（转载）\n一旦没有了"),
    ("我都退伍2014-10-30 20:10 上传 好久了", "我都退伍 好久了"),
    ("值得深思！2024-05-28 23:09 来自浙江省\n查看回复", "值得深思！\n查看回复"),
    ("2017-04-18 23:01:00\n今天", "今天"),
    ("现在随意化。 2018.8.22日 12:27:44\n2.现在版面上的有", "现在随意化。\n2.现在版面上的有"),
]

# Texts whose redaction once must stay as it is when redacted again.
AGAIN = [
    "群里@所有人今天下午三点开会请大家准时到场不要迟到谢谢配合辛苦了各位同事们",
    "@张三 @李四 你们看一下",
    "回复@联合光子a: 说得对",
    "联系人：张三 电话13812345678 微信号：abc123 邮箱a@b.com",
]


def main():
    failures = 0
    for text, expected in REDACT:
        got = redact(text)
        if got != expected:
            failures += 1
            print(f"FAIL redact {text!r}: {got!r}, expected {expected!r}")
    for text in KEEP:
        got = clean(text)
        if got != text:
            failures += 1
            print(f"FAIL keep {text!r}: {got!r}")
    for text, expected in CLEAN:
        got = clean(text)
        if got != expected:
            failures += 1
            print(f"FAIL clean {text!r}: {got!r}, expected {expected!r}")
    # Running the rules again changes nothing: the learning store redacts its records at every start.
    for text in [t for t, _ in REDACT] + KEEP + [t for t, _ in CLEAN] + AGAIN:
        once = redact(text)
        if redact(once) != once:
            failures += 1
            print(f"FAIL twice {text!r}: {once!r} → {redact(once)!r}")
    total = len(REDACT) + len(KEEP) + len(CLEAN) + len(AGAIN)
    print("ALL OK" if not failures else f"{failures} of {total} failed", f"({total} cases)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
