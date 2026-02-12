import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

# --- CONFIGURACIÓN ---
GMAIL_USER = "juan.rincon@premexcorp.com"
GMAIL_APP_PASSWORD = "aohfbmjmegzagtac"
DESTINATARIO = "jdrincone@gmail.com"

# El archivo que generamos anteriormente
ARCHIVO_A_ENVIAR = "informe_cmi_microbiology.html" 

def enviar_correo_estilo_okuo(archivo_path):
    # Verificar si el archivo existe
    if not os.path.exists(archivo_path):
        print(f"❌ Error: El archivo '{archivo_path}' no existe en la carpeta actual.")
        return

    # Obtener fecha actual para el reporte
    fecha_hoy = datetime.now().strftime("%d-%m-%Y")
    nombre_archivo = os.path.basename(archivo_path)

    # Crear el mensaje
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"Okuo Analytics: Reporte Microbiológico - {fecha_hoy}"
    msg['From'] = f"Okuo Analytics <{GMAIL_USER}>"
    msg['To'] = DESTINATARIO

    # --- DISEÑO HTML (Estilo Okuo Analytics) ---
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Reporte Okuo Analytics</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #333333; margin: 0; padding: 0; background-color: #f4f4f4; }}
            .container {{ max-width: 650px; margin: 20px auto; background-color: #ffffff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
            .header {{ text-align: center; margin-bottom: 20px; }}
            .brand {{ font-size: 32px; font-weight: bold; color: #374151; }}
            .brand span {{ color: #1C8074; }} /* El verde de Okuo */
            .subtitle {{ font-size: 11px; letter-spacing: 3px; color: #9CA3AF; text-transform: uppercase; margin-top: 5px; font-weight: bold; }}
            .divider {{ height: 3px; background-color: #1C8074; width: 100%; margin-top: 20px; margin-bottom: 30px; border: none; }}
            .content {{ font-size: 15px; line-height: 1.6; color: #4B5563; }}
            
            /* LA CAJA DE RESUMEN (Estilo de tu imagen) */
            .summary-box {{
                background-color: #F9FAFB;
                border-left: 5px solid #1C8074; /* Borde verde grueso a la izquierda */
                padding: 20px;
                margin: 25px 0;
                border-radius: 4px;
            }}
            .summary-title {{
                color: #1C8074;
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 10px;
                display: block;
                border-left: 0; /* Reset para el texto */
            }}
            .file-item {{
                font-weight: 500;
                color: #111827;
                list-style-type: none;
            }}
            
            .footer {{ margin-top: 40px; text-align: center; font-size: 11px; color: #9CA3AF; border-top: 1px solid #E5E7EB; padding-top: 20px; }}
            .link-green {{ color: #1C8074; text-decoration: none; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="brand">Okuo <span>Analytics</span></div>
                <div class="subtitle">Intelligent Data Solutions</div>
            </div>
            
            <hr class="divider">

            <div class="content">
                <p>Estimado equipo,</p>
                
                <p>Se adjunta el <strong>Diagnóstico Microbiológico Integral</strong> generado automáticamente por la plataforma <strong>Okuo Analytics</strong>.</p>
                
                <p>Este documento contiene el análisis de eficacia del tratamiento en maíz, la evaluación de proveedores y la detección de puntos de recontaminación en planta correspondientes al periodo de <strong>Enero 2026</strong>.</p>

                <div class="summary-box">
                    <span class="summary-title">Archivo Adjunto Disponible:</span>
                    <ul style="padding-left: 20px; margin: 0;">
                        <li class="file-item">📄 {nombre_archivo}</li>
                    </ul>
                </div>

                <p>Para revisar el reporte completo, descarga el archivo adjunto y ábrelo desde tu navegador web.</p>
            </div>
    
            <div class="footer">
                Este es un correo automático generado por el sistema de reportes.<br>
                &copy; 2026 Okuo Analytics · <a href="https://okuo.bio" class="link-green">okuo.bio</a>
            </div>
        </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(html_content, 'html'))

    # --- ADJUNTAR EL ARCHIVO ---
    try:
        with open(archivo_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {nombre_archivo}",
        )
        msg.attach(part)
    except Exception as e:
        print(f"❌ Error al adjuntar el archivo: {e}")
        return

    # --- ENVIAR CORREO ---
    try:
        print("Conectando con el servidor SMTP...")
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print(f"✅ Reporte enviado exitosamente a: {DESTINATARIO}")
    except Exception as e:
        print(f"❌ Error al enviar el correo: {e}")

if __name__ == "__main__":
    enviar_correo_estilo_okuo(ARCHIVO_A_ENVIAR)