from PIL import Image

# 打开背景和头部图像
background = Image.open("background.png")
head = Image.open("head.png")

# 获取背景的宽高
bg_width, bg_height = background.size

# 获取头部图像的宽高
head_width, head_height = head.size

# 计算头部图像的位置（将头部图像放置在背景图像的中央）
position = ((bg_width - head_width) // 2, (bg_height - head_height) // 2)

# 将头部图像粘贴到背景上，保留透明度（如果有的话）
background.paste(head, position, head)

# 保存结果图像
background.save("output.png")

# 显示结果
background.show()

