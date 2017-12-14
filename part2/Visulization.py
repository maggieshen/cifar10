########################################################################
# The functions are for showing the original images
########################################################################
def plot_images(images, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
            
        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


########################################################################
# The functions are for showing different layer for the networks
########################################################################
def getActivations(sess,layer_name,image_name):
    units = sess.run(layer_name,feed_dict={x:image_name})
    plotNNFilter(units)
    
    
def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

#######################################################
# The function is for obtaining hard assignment for the images and show the confusion matrix
#######################################################
def Test_Anlaysis(sess, images, labels, cls_true):
    
    num_images = len(images)
    # Initialize the predictions
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    
    i = 0
    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)
        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = sess.run(prediction, feed_dict={x: images[i:j, :],y_true: labels[i:j, :]})

        # end-index of the current batch.
        i = j
    
    co_ma = confusion_matrix(y_true=cls_true, y_pred=cls_pred) 
    
    # Print the confusion matrix as text.
    print("Confusion Matrix as numbers:")  
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(co_ma[i, :], class_name)
    
    # Print the consusion matrix as graph
    print("Plot Confusion Matrix:")  
    df_cm = pd.DataFrame(co_ma, index = [i for i in class_names],
                  columns = [i for i in class_names])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
