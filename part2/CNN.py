#######################################################
# The function is for compute the cross entropy
#######################################################
def compute_cross_entropy(logits, y):
    sm_ce = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits, name='cross_ent_terms')
    cross_ent = tf.reduce_mean(sm_ce, name='cross_ent')
    return cross_ent


#######################################################
# The function is for compute the cross entropy
#######################################################
def compute_accuracy(logits, y):
    prediction = tf.argmax(logits, 1, name='pred_class')
    true_label = tf.argmax(y, 1, name='true_class')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, true_label), tf.float32))
    return accuracy

#######################################################
# The function is for geting random batch for training the network
#######################################################
def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)
    
    # Create a random index.
    idx = np.random.choice(num_images,
                           size=batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx,:,:,:]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch

######################################################
# The Function for defineing the CNN Network
######################################################