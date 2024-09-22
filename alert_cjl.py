import requests

headers = {
    "Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjQ1MTE2MywidXVpZCI6ImU4NzE3ODQwLWRmMzUtNDE2Mi05ZDNlLTJjMTYwYmJjYjA2YSIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.SGnD3iCAUTGwI3lU6AH-Ktdw2Z1K72dhK525eJaEy0oRkA0HjGk3flGb5_RO4svKxeyzrR2pxmOjRIBNeIxo5g"
}
resp = requests.post(
    "https://www.autodl.com/api/v1/wechat/message/send",
    json={
        "title": "训练结束力",
        "name": "Coviar炼丹结束力",
        "content": "114514",
    },
    headers=headers,
)
print(resp.content.decode())
