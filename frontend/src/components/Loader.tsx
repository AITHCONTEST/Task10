
import styles from "./styles/loader.module.scss";
import loaderGif from "./../assets/loading-50.gif";

const Loader = () => {
    return(
        <div className={styles.loader}>
            <div className={styles.loader__imgContainer}>
                <img className={styles.loader__img} src={loaderGif} alt="..." />
            </div>
        </div>
    );
};

export default Loader;