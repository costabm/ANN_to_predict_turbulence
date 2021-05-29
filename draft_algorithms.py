# # HOW TO TRANSFORM STANDARD DEVIATIONS FROM ONE SYSTEM TO ANOTHER (COVARIANCE MATRIX REQUIRED)
# from read_0p1sec_data import R_z
# import numpy as np
# V_1 = np.array([[8,12,14,20], [11,9,10,10], [-1, 0, 2,1]])
# T_21 = R_z(np.pi/4).T
# V_2 = T_21 @ V_1
# V_1_stds = np.std(V_1, axis=1)
# V_1_cov = np.cov(V_1, bias=True)
# V_2_stds = np.std(V_2, axis=1)
# V_2_stds_2nd_meth
#
# od = np.sqrt(np.diag(T_21 @ V_1_cov @ T_21.T))






server.login('djbernardocosta@hotmail.com', "444382")
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()





import win32com.client as client
outlook = client.Dispatch("Outlook.Application")
message = outlook.CreateItem(0)
message.Display()
message.To = 'bernamdc@gmail.com'
message.CC = ''
message.BCC = ''

message.Subject = 'Teste benny'
message.Body = 'Conseguimos'






import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
mail_content = '''Olá Bernardo, será que isto funciona?
'''
#The mail addresses and password
sender_address = 'djbernardocosta@hotmail.com'
sender_pass = '444382'
receiver_address = 'bernamdc@gmail.com'
#Setup the MIME
message = MIMEMultipart()
message['From'] = sender_address
message['To'] = receiver_address
message['Subject'] = 'A test mail sent by Python.'   #The subject line
#The body and the attachments for the mail
message.attach(MIMEText(mail_content, 'plain'))
#Create SMTP session for sending the mail
session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
session.starttls() #enable security
session.login(sender_address, sender_pass) #login with mail_id and password
text = message.as_string()
session.sendmail(sender_address, receiver_address, text)
session.quit()
print('Mail Sent')