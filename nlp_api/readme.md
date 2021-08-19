# 启动
python ./manager.py runserver



# 根据数据库表去生成数据model
flask-sqlacodegen mysql+pymysql://root:f408909685@115.28.79.206:3306/gaokao?charset=utf8 --tables blog --outfile "common/models/Blog.py"  --flask

flask-sqlacodegen mysql+pymysql://xzh:Xing123456@rm-2zeh8v6x35m6prfem3o.mysql.rds.aliyuncs.com:3306/gaokao?charset=utf8 --tables school_subject --outfile "common/models/SchoolSubject.py"  --flask