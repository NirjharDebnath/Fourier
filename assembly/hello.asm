section .data
    msg db "Hello, World!\n", 0Ah

section .text
    global _start

_start:
    mov edx, 14
    mov ecx, msg
    mov ebx, 1
    mov eax, 4
    int 0x80

    mov eax, 1
    xor ebx, ebx
    int 0x80
