# Reference only - PythonAnywhere generates its own WSGI file at:
#   /var/www/<your_username>_pythonanywhere_com_wsgi.py
# Open that file in the PythonAnywhere "Web" tab editor and replace its
# contents with this, updating <your_username> to your actual username.

import sys

path = '/home/<your_username>/spam-classifier-ai/ai'
if path not in sys.path:
    sys.path.append(path)

from app import app as application
