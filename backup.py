"""File backup

"""

# Important Files Backup
import os
import time
import zipfile

source = [r'C:\Users\mingle\Desktop\backup.txt']    # Notice we had to use double quotes inside the string for names with spaces in it.
target_dir = r'D:\Backup'

if not os.path.exists(target_dir):
    os.mkdir(target_dir)
    print('Successfully created directory', target_dir)
today = target_dir + os.sep + time.strftime('%Y%m%d')
now = time.strftime('%H%M%S')

comment = input('Enter a comment --> ')
if len(comment) == 0:
    target = today + os.sep + now + '.zip'
else:
    target = today + os.sep + now + '_' + comment.replace(' ', '_') + '.zip'

if not os.path.exists(today):
    os.mkdir(today)
    print('Successfully created directory', today)

with zipfile.ZipFile(target, 'w') as backupfile:
    for file in source:
        backupfile.write(file)