# Sending gmail using SMTP_SSL()
# using port 465

import smtplib, ssl

import getpass
import math, random 


# function to send email
def sendEmail(otp):
	sender_email = "072bct535@pcampus.edu.np"
	receiver_email = "shishirbhandari54@gmail.com"
	message = """\
	Subject: Speaker Identification Authentication

	Your OTP is: """

	message += otp

	port = 465 # for ssl
	password = getpass.getpass(prompt="Type gmail password: ")


	#create a secure SSL context
	context = ssl.create_default_context()

	with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
		server.login("072bct535@pcampus.edu.np", password)
		#sending email
		server.sendmail(sender_email, receiver_email, message)




# function to generate OTP 
def generateOTP() : 
  
    # Declare a string variable   
    # which stores all characters  
    string = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    OTP = "" 
    length = len(string) 
    for i in range(6) : 
        OTP += string[math.floor(random.random() * length)] 
  
    return OTP