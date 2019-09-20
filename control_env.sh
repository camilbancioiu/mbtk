#!/usr/bin/env bash
systemctl stop NetworkManager ModemManager bluetooth wpa_supplicant avahi-daemon firewalld
systemctl stop ccpd colord
systemctl stop httpd php-fpm mysqld postgresql memcached
systemctl stop dbus abrtd cups
systemctl stop radicale
systemctl stop pulseaudio
