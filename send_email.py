from urllib.request import urlopen
import smtplib
from email.message import EmailMessage
import time

password = '444382'

def send_mail(to_email, subject, message, from_email):
    # import smtplib
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = ', '.join(to_email)
    msg.set_content(message)
    print(msg)
    server = smtplib.SMTP('smtp.office365.com', 587)
    server.connect("smtp.office365.com", 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.set_debuglevel(1)
    server.login(from_email, password)  # user & password
    server.send_message(msg)
    server.quit()
    print('successfully sent the mail.')


while True:
    link = 'https://covid19.min-saude.pt/pedido-de-agendamento/'
    f = urlopen(link)
    myfile = f.read()

    if not '50 ou mais anos' in str(myfile):
        send_mail(to_email=['bernamdc@gmail.com'], subject='Hora de ir ver o site do SNS',
                  message='A idade no site do SNS deve ter mudado, vai la ver.',
                  from_email='djbernardocosta@hotmail.com')
        send_mail(to_email=['madalenafurtado23@gmail.com'], subject='Hora de ir ver o site do SNS',
                  message='A idade no site do SNS deve ter mudado, vai la ver.',
                  from_email='djbernardocosta@hotmail.com')
        time.sleep(300)  # tem la calma viotty!!
        send_mail(to_email=['luis.viotty@gmail.com'], subject='Hora de ir ver o site do SNS',
                  message='A idade no site do SNS deve ter mudado, vai la ver.',
                  from_email='djbernardocosta@hotmail.com')
        print('email sent!!!')
        break
    time.sleep(30)





# # Testing with another website
# while True:
#     link = 'https://www.timeanddate.com/worldclock/norway/oslo'
#     f = urlopen(link)
#     myfile = f.read()
#
#     if '12:22' in str(myfile):
#         send_mail(to_email=['bernamdc@gmail.com'], subject='teste', message='Olá de novo',
#                   from_email='djbernardocosta@hotmail.com')
#         print('email sent!!!')
#         break
#     time.sleep(10)


