import matplotlib.pyplot as plt
from main import load_data, your_images_loader

def plot_img_points(ax, img, points_mat):
    ax.imshow(img)
    ax.scatter(points_mat[0], points_mat[1], marker='.', color='r')

def plot_assignment_imgs():
    for (matching,title) in zip([True, False], ['perfect matching','non perfect matching']):
        # loading data with perfect matches
        src_img, dst_img, match_p_src, match_p_dst = load_data(is_perfect_matches=matching)

        f,ax=plt.subplots(1,2, figsize=(8,4))
        for i, (img,points_mat) in enumerate(zip([src_img, dst_img], [match_p_src, match_p_dst])):
            plot_img_points(ax[i], img, points_mat)
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.savefig(title)
        plt.show()

def plot_test_imgs():
    src_img, dst_img, match_p_src, match_p_dst = your_images_loader()
    f, ax = plt.subplots(1, 2, figsize=(8, 4))
    for i, (img, points_mat) in enumerate(zip([src_img, dst_img], [match_p_src, match_p_dst])):
        plot_img_points(ax[i], img, points_mat)
    plt.suptitle('test imgs matching points')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig('test imgs matching points')
    plt.show()


def main():
    plot_assignment_imgs()
    plot_test_imgs()



if __name__ == '__main__':
    main()
