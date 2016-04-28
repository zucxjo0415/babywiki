cd BasicBrowser
python manage.py syncdb
mv tmv_db ..
cd ..
python ldafeed.py 
