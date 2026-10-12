@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 같은 와이파이의 다른 기기도 접속할 수 있게 서버를 켭니다.
echo 아래에 나오는 http://192.168.x.x:8810/ 주소를 공유하세요. (끄려면 이 창에서 Ctrl+C)
echo.
start "" http://localhost:8810/
python server.py --lan
