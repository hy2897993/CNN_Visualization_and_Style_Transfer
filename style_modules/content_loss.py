import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def forward(self, content_weight, content_current, content_original):
        """
            Compute the content loss for style transfer.

            Inputs:
            - content_weight: Scalar giving the weighting for the content loss.
            - content_current: features of the current image; this is a PyTorch Tensor of shape
              (1, C_l, H_l, W_l).
            - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

            Returns:
            - scalar content loss
            """

        ##############################################################################
        # TODO: Implement content loss function                                      #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################
        # Lc = wc * (F-P)**2
        
        P = torch.squeeze(content_original) #  C, H, W
        F = torch.squeeze(content_current) #  C, H, W
        
        return content_weight*torch.sum((P-F)**2) 
    



        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

