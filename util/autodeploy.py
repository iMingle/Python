# 配置参数
ip = '127.0.0.1' # 部署服务器地址
remote_dir = '/opt/project/' # 部署服务器项目目录
local_dir = './project/target/' # 本地打包文件的目录
zipfile = 'project.zip' # 本地打包的zip文件

if __name__ == '__main__':
    import paramiko
    import os

    if os.system('mvn clean package') is 0:
        try:
            username = 'username'
            password = 'password'
            port = 22

            #生成ssh客户端实例
            s = paramiko.SSHClient()
            s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            print("starting deploy")
            s.connect(ip, port, username, password)
            stdin, stdout, stderr = s.exec_command('rm -rf ' + remote_dir + zipfile[0:zipfile.rindex('.')] + '*')
            stdout.read()
            stdin, stdout, stderr = s.exec_command('rm -f ' + remote_dir + zipfile)
            stdout.read()

            stdin, stdout, stderr = s.exec_command('ps ax | grep ' + zipfile[0:zipfile.rindex('.')])
            process = str(stdout.read(), encoding = "utf-8")
            for line in process.split('\n'):
                if line.find(remote_dir + zipfile[0:zipfile.rindex('.')]) != -1:
                    arr = line.split(' ')
                    stdin, stdout, stderr = s.exec_command('kill -9 ' + arr[0])
                    stdout.read()
            
            transport = paramiko.Transport((ip, port))
            transport.connect(username = username, password = password)
            sftp = paramiko.SFTPClient.from_transport(transport)

            # 将文件从windows系统拷贝到linux系统执行目录
            sftp.put(local_dir + zipfile, remote_dir + zipfile)
            sftp.close()
            transport.close()

            stdin, stdout, stderr = s.exec_command('unzip ' + remote_dir + zipfile + ' -d ' + remote_dir)
            stdout.read()

            stdin, stdout, stderr = s.exec_command('dos2unix ' + remote_dir + zipfile[0:zipfile.rindex('.')] + '/bin/*.sh')
            stdout.read()
            stdin, stdout, stderr = s.exec_command('chmod u+x ' + remote_dir + zipfile[0:zipfile.rindex('.')] + '/bin/*.sh')
            stdout.read()
            stdin, stdout, stderr = s.exec_command('/bin/bash ' + remote_dir + zipfile[0:zipfile.rindex('.')] + '/bin/start.sh &')
            print("deploy successfully")
            stdout.read()

            s.close()
        except Exception as e:
            print(e)
