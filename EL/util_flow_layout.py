# refer from: https://stackoverflow.com/questions/21754976/ipython-notebook-arrange-plots-horizontally

import matplotlib.pyplot as plt
from IPython.display import HTML, display

import io
import base64


class FlowLayout(object):
    ''' A class / object to display plots in a horizontal / flow layout below a cell '''
    def __init__(self):
        # string buffer for the HTML: initially some CSS; images to be appended
        self.sHtml =  """
        <style>
        .floating-box {
        display: inline-block;
        margin: 10px;
        border: 3px solid #888888;  
        }
        </style>
        """

    def add_plot(self, oAxes):
        ''' Saves a PNG representation of a Matplotlib Axes object '''
        Bio=io.BytesIO() # bytes buffer for the plot
        fig = oAxes.get_figure()
        fig.canvas.print_png(Bio) # make a png of the plot in the buffer

        # encode the bytes as string using base 64 
        sB64Img = base64.b64encode(Bio.getvalue()).decode()
        self.sHtml+= (
            '<div class="floating-box">'+ 
            '<img src="data:image/png;base64,{}\n">'.format(sB64Img)+
            '</div>')

    def PassHtmlToCell(self):
        ''' Final step - display the accumulated HTML '''
        display(HTML(self.sHtml))

    def draw(self, axes):
        for ax in axes:
            self.add_plot(ax)
            plt.close()
        self.PassHtmlToCell()


if __name__ == "__main__":
    import numpy as np


    oPlot = FlowLayout() # create an empty FlowLayout

    # Some fairly regular plotting from Matplotlib
    gX = np.linspace(-5,5,100) # just used in the plot example
    for i in range(10): # plot 10 charts
        fig, ax = plt.subplots(1, 1, figsize=(3,2)) # same size plots
                            # figsize=(3+i/3,2+i/4)) # different size plots
        ax.plot(gX, np.sin(gX*i)) # make your plot here
        oPlot.add_plot(ax) # pass it to the FlowLayout to save as an image
        plt.close() # this gets rid of the plot so it doesn't appear in the cell

    # import ipdb; ipdb.set_trace()

    oPlot.PassHtmlToCell()
