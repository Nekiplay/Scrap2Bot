#!/bin/bash

# Скрипт для подключения к Android по Wi-Fi (с обработкой USB-подключения)

# Параметры
IP="10.10.10.185"
PORT="5555"
SCRCPY_OPTIONS="-e --video-bit-rate 2M --video-codec h264 --max-size 800 --max-fps=60 --disable-screensaver --keyboard=disabled"

# Функции
check_wifi_connection() {
    echo "1. Проверка Wi-Fi подключения к $IP:$PORT..."
    if ping -c 1 $IP > /dev/null 2>&1; then
        echo "✓ Устройство доступно в сети"
        return 0
    else
        echo "ОШИБКА: Устройство не ответило на ping"
        echo "Убедитесь что:"
        echo "1. Устройство и компьютер в одной сети Wi-Fi"
        echo "2. На устройстве включена отладка по Wi-Fi"
        echo "3. IP-адрес указан верно (текущий: $IP)"
        return 1
    fi
}

disconnect_usb_if_connected() {
    echo "1.5 Проверка USB-подключения..."
    USB_DEVICE=$(adb devices | grep -v "List of devices" | grep -v "$IP:$PORT" | grep "device$" | awk '{print $1}')
    
    if [ -n "$USB_DEVICE" ]; then
        echo "⚠ Обнаружено USB-подключение: $USB_DEVICE"
        echo "Отключение USB-подключения для перехода на Wi-Fi..."
        adb -s $USB_DEVICE tcpip $PORT
        sleep 3
        return 0
    fi
    return 0
}

connect_device() {
    echo "2. Подключение к устройству..."
    adb disconnect "$IP:$PORT" > /dev/null 2>&1
    CONNECT_RESULT=$(adb connect "$IP:$PORT")
    
    if echo "$CONNECT_RESULT" | grep -q "connected"; then
        echo "✓ $CONNECT_RESULT"
        return 0
    elif echo "$CONNECT_RESULT" | grep -q "already connected"; then
        echo "⚠ $CONNECT_RESULT"
        return 0
    else
        echo "ОШИБКА: $CONNECT_RESULT"
        return 1
    fi
}

verify_connection() {
    echo "3. Проверка состояния устройства..."
    local state=$(adb devices | grep "$IP:$PORT" | awk '{print $2}')
    
    case $state in
        "device") 
            echo "✓ Устройство готово к работе" 
            return 0 
            ;;
        "offline") 
            echo "ОШИБКА: Устройство offline"
            echo "Попробуйте:"
            echo "1. Перезагрузить устройство"
            echo "2. Отключить и снова включить отладку по Wi-Fi"
            return 1 
            ;;
        "unauthorized") 
            echo "ОШИБКА: Не авторизовано"
            echo "Проверьте разрешение отладки на устройстве"
            return 1 
            ;;
        *) 
            echo "ОШИБКА: Не удалось подключиться (состояние: $state)"
            return 1 
            ;;
    esac
}

# Основной скрипт
echo "=== Wi-Fi отладка Android (беспроводной режим) ==="

# Шаг 1: Проверка подключения к устройству в сети
check_wifi_connection || exit 1

# Шаг 1.5: Отключение USB если подключено
disconnect_usb_if_connected

# Шаг 2: Подключение ADB
connect_device || {
    echo "Попытка решения: перезапуск ADB сервера..."
    adb kill-server
    adb start-server
    sleep 2
    connect_device || exit 1
}

# Шаг 3: Проверка состояния подключения
verify_connection || {
    echo "Попытка решения: сброс соединения..."
    adb disconnect "$IP:$PORT"
    sleep 2
    adb connect "$IP:$PORT"
    verify_connection || exit 1
}

# Запуск scrcpy
echo "4. Запуск scrcpy..."
scrcpy $SCRCPY_OPTIONS || {
    echo "ОШИБКА: Не удалось запустить scrcpy"
    echo "Проверьте установку scrcpy:"
    exit 1
}

# Завершение
adb disconnect "$IP:$PORT"
echo "=== Готово ==="
