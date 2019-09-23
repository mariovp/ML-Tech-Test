import pika
import time 

credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', credentials=credentials, port=32775))
channel = connection.channel()

for method_frame, properties, body in channel.consume('test'):
    # Display the message parts and acknowledge the message
    print(method_frame, properties, body)
    print("Training network with message...")
    time.sleep(30)
    print("Ready!")
    channel.basic_ack(method_frame.delivery_tag)

    # Escape out of the loop after 10 messages
    if method_frame.delivery_tag == 10:
        break

# Cancel the consumer and return any pending messages
requeued_messages = channel.cancel()
print('Requeued %i messages' % requeued_messages)
connection.close()