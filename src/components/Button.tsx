import styles from "./styles/button.module.scss";


interface ButtonProps {
    img: string;
    tap: () => void;
    alt: string
}

const Button = ({ img, tap, alt}: ButtonProps) => {
    return (
        <div onClick={tap} className={styles.butt}>
            <img src={img} alt={alt} />
        </div>
    );
};

export default Button;
