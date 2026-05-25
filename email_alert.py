import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


def send_email_alert(event_name, timestamp, image_path):

    sender_email = "SENDER_MAIL"
    receiver_email = "RECEIVER_MAIL"
    app_password = "YOUR_PASSWORD"

    subject = f"🚨 Alert: {event_name} Detected"

    body = f"""
AI Home Security Alert

Event: {event_name}
Time: {timestamp}

Please check the attached snapshot.
"""

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    # Attach snapshot image
    with open(image_path, "rb") as img:
        image = MIMEImage(img.read())
        msg.attach(image)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully")

    except Exception as e:
        print("Email failed:", e)