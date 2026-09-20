import matplotlib.pyplot as plt

def plot_two_grids(data1, data2, title1="Plot 1", title2="Plot 2"):
    """
    將兩個圖表顯示在同一個視窗中（1,2）
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.imshow(data1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.imshow(data2, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def plot_image_and_hist_grid(img_before, img_after, title_before="Before", title_after="After"):
    """
    影像與直方圖的對比 (2,2)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 影像部分 (row 1)
    axes[0, 0].imshow(img_before, cmap='gray')
    axes[0, 0].set_title(title_before)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_after, cmap='gray')
    axes[0, 1].set_title(title_after)
    axes[0, 1].axis('off')
    
    # 直方圖部分 (row 2)
    # ravel() 將二維矩陣攤平為一維，以便繪製頻率分佈
    axes[1, 0].hist(img_before.ravel(), bins=256, range=[0, 256], color='black')
    axes[1, 0].set_title(title_before)
    axes[1, 0].set_xlim([0, 256])
    
    axes[1, 1].hist(img_after.ravel(), bins=256, range=[0, 256], color='black')
    axes[1, 1].set_title(title_after)
    axes[1, 1].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.show()