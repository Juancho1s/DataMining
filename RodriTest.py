import serial
import time

# Cambia la ruta al puerto correcto del módulo Bluetooth
bluetooth_port = "COM18"  # Actualiza esto con el puerto correcto
baud_rate = 9600

try:
    # Establecer conexión con el módulo Bluetooth
    bluetooth = serial.Serial(bluetooth_port, baud_rate, timeout=1)
    print("Conexión establecida con el módulo Bluetooth.")
    time.sleep(2)  # Espera para estabilizar la conexión

    while True:
        # Enviar un mensaje al Arduino a través del Bluetooth
        mensaje = input("Escribe un mensaje para el Arduino: ")
        bluetooth.write((mensaje + "\n").encode())  # Enviar mensaje con nueva línea

        # Leer la respuesta del Arduino
        time.sleep(0.5)  # Esperar un poco para que llegue la respuesta
        if bluetooth.in_waiting > 0:
            data = bluetooth.read(bluetooth.in_waiting).decode(errors='ignore')
            print(f"Respuesta del Arduino: {data}")

except KeyboardInterrupt:
    print("Programa interrumpido por el usuario.")

except serial.SerialException as e:
    print(f"Error de conexión: {e}")

finally:
    if 'bluetooth' in locals() and bluetooth.is_open:
        bluetooth.close()
        print("Conexión cerrada.")