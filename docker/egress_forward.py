#!/usr/bin/env python3
"""TCP-форвардер с одним назначением.

Слушает ``--listen host:port`` и каждый принятый сокет соединяет только с
``--target host:port``. Полезная нагрузка не разбирается: HTTP CONNECT и любой
другой адрес внутри байтов назначение не меняют.
"""

from __future__ import annotations

import argparse
import socket
import threading


def _parse_hostport(text: str) -> tuple[str, int]:
    host, sep, port = text.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise SystemExit(f"expected host:port, got {text!r}")
    number = int(port)
    if not 0 < number < 65536:
        raise SystemExit(f"port out of range: {text!r}")
    return host, number


def _splice(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _handle(client: socket.socket, target: tuple[str, int]) -> None:
    """Соединиться только с *target*. Другого ``connect`` в процессе нет."""
    remote: socket.socket | None = None
    try:
        remote = socket.create_connection(target, timeout=10)
        left = threading.Thread(target=_splice, args=(client, remote), daemon=True)
        right = threading.Thread(target=_splice, args=(remote, client), daemon=True)
        left.start()
        right.start()
        left.join()
        right.join()
    except OSError:
        pass
    finally:
        client.close()
        if remote is not None:
            remote.close()


def serve(listen: str, target: tuple[str, int]) -> None:
    host, port = _parse_hostport(listen)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)
    try:
        while True:
            client, _addr = sock.accept()
            threading.Thread(target=_handle, args=(client, target), daemon=True).start()
    finally:
        sock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forward TCP to one configured host:port and nowhere else.",
    )
    parser.add_argument("--listen", required=True, help="local host:port")
    parser.add_argument("--target", required=True, help="the only host:port to connect to")
    args = parser.parse_args(argv)
    serve(args.listen, _parse_hostport(args.target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
