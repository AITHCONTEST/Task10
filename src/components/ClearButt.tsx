import clearImg from "./../assets/new/clear.svg";
import Button from "./Button";
import styles from "./styles/buttons/clear.module.scss";

interface ClearButtProps {
    tap: ()=>void;
}

const ClearButt = ({ tap }: ClearButtProps) => {

    return (
        <div className={styles.clear}>
            <Button img={clearImg} tap={tap}  alt={"clear"}/>
        </div>
        
    );
};

export default ClearButt;
