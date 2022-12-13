import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from main import load_data

def plot_img_points(ax, img, points_mat):
    ax.imshow(img)
    ax.scatter(points_mat[0], points_mat[1], marker='.', color='r')

def main():

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



if __name__ == '__main__':
    main()
