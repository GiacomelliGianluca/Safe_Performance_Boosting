import argparse

# argument parser for plotting
def plt_parser():
    parser = argparse.ArgumentParser(description="Plot settings.")

    # Font dimensions
    parser.add_argument('--font_dim', type=float, default=12.0, help='Dimension of the font for the xlabel and ylabel. Default is 12.')
    parser.add_argument('--title_font_dim', type=float, default=12.0, help='Dimension of the font for the title. Default is 12.')
    # Font type
    parser.add_argument('--fontname', type=str, default='Times New Roman', help='Utilized font type. Default is Times')
    # Line Thickness
    parser.add_argument('--thick', type=float, default=1.6, help='Line thickness. Default is 1.6.')
    # Grid
    parser.add_argument('--grid', type=bool, default= True, help='Grid. Default is True')
    # Plot document
    parser.add_argument('--document', type=str, default='paper', help='Type of document. Default is paper.')


    args = parser.parse_args()

    if args.document == "paper": # TODO: Chiedere a Luca/Danilo la dimensione delle figure e il font che usano
        args.plot_dim_x = 3.15 # (inch). Dimension of the x-axis
        args.plot_dim_y = 2.36 # (inch). Dimension of the y-axis

    if args.document == "ppt":
        args.plot_dim_x = 12.0 # (inch). Dimension of the x-axis
        args.plot_dim_y = 5.0 # (inch). Dimension of the y-axis




    return args


