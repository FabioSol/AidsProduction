import os
print(os.listdir(os.path.dirname(__file__)))

from api.main import app

if __name__ == '__main__':
    app.run(port=8080)