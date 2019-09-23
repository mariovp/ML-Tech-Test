import pika

credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', credentials=credentials))
channel = connection.channel()
channel.queue_declare(queue='test')
channel.basic_publish(exchange='', routing_key='test',
                      body=b'Test message.')
print("Successfully published message")
connection.close()